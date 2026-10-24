"""Evaluate a trained model on val / internal test / locked external test and report the proposal's metrics.
The external set is only ever *evaluated*; the decision threshold comes from the VALIDATION split of the dev site."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import data as D
from model import ReportModel

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--ext", choices=["busbra", "breast"]); p.add_argument("--ext_root")
p.add_argument("--bs", type=int, default=32); p.add_argument("--boot", type=int, default=1000)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
run = Path(a.run)
ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
ar, schema = ck["args"], ck["schema"]; D.register_values(schema)
tok = D.Tok(texts=ck["tok"]["texts"], hf=ck["tok"]["hf"], max_len=ck["tok"]["max_len"])
full = ar["model"] == "breast_vlreport"
enc = {"breast_vlreport": "swin", "swin_transformer": "swin", "vit_transformer": "vit"}.get(ar["model"], "cnn")
dec = ar["decoder"] or ("lstm" if ar["model"] == "cnn_lstm" else "tfm")
model = ReportModel(tok, schema, enc, dec, lesion=full and not ar["no_lesion"], attr=full and not ar["no_lesion"] and not ar["no_attr"],
                    contrastive=full and not ar["no_contrastive"], pretrained=False)
model.load_state_dict(ck["state"]); model.to(dev).eval()
rng = np.random.default_rng(0)


@torch.no_grad()
def predict(df):
    dl = DataLoader(D.BUSData(df, schema, tok), a.bs, num_workers=2)
    texts, dice, attr, prob = [], [], {k: [] for k in schema}, []
    pi = schema["pathology"].index("malignant") if "malignant" in schema.get("pathology", []) else None
    for b in dl:
        ids, o = model.generate(b["img"].to(dev))
        texts += [tok.decode(r) for r in ids.cpu()]
        if "seg" in o:
            pr, g = (torch.sigmoid(o["seg"]) > .5).float().cpu(), b["mask"]
            dice += ((2 * (pr * g).sum((1, 2, 3)) + 1e-6) / (pr.sum((1, 2, 3)) + g.sum((1, 2, 3)) + 1e-6)).tolist()
        if "attr" in o:
            for k in schema: attr[k] += o["attr"][k].argmax(-1).cpu().tolist()
            prob += torch.softmax(o["attr"]["pathology"], -1)[:, pi].cpu().tolist()
    return dict(texts=texts, dice=np.array(dice), attr=attr, prob=np.array(prob))


def ci(x, fn):
    x = np.asarray(x); v = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(a.boot)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def clinical(texts, df):
    """Facts stated in the generated report vs. the structured ground truth (template-level, NOT radiologist-level)."""
    names = [k for k in schema]
    stated = wrong = gt_total = missing = bad_reports = 0; per = []
    for t, at in zip(texts, df.att):
        pr = D.parse_report(t, names)
        s = w = 0
        for k, v in pr.items():
            if k in at: s += 1; w += int(v != at[k])        # only facts verifiable against GT
        gt_total += len([k for k in at if k in names]); missing += len([k for k in at if k in names and k not in pr])
        stated += s; wrong += w; bad_reports += int(w > 0); per.append((s, w))
    per = np.array(per)
    return dict(factual_error_rate=wrong / max(stated, 1), hallucination_report_rate=bad_reports / len(texts),
                omission_rate=missing / max(gt_total, 1),
                factual_error_rate_ci=ci(np.arange(len(per)), lambda i: per[i, 1].sum() / max(per[i, 0].sum(), 1))), per


def nlp(texts, df):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    gts = {i: [D.norm_text(D.build_report(x))] for i, x in enumerate(df.att)}
    res = {i: [t if t else "empty"] for i, t in enumerate(texts)}
    r = {}
    b, _ = Bleu(4).compute_score(gts, res, verbose=0); r.update({f"BLEU-{i+1}": float(v) for i, v in enumerate(b)})
    r["ROUGE-L"] = float(Rouge().compute_score(gts, res)[0]); r["CIDEr"] = float(Cider().compute_score(gts, res)[0])
    try:
        from nltk.translate.meteor_score import meteor_score
        r["METEOR"] = float(np.mean([meteor_score([gts[i][0].split()], res[i][0].split()) for i in gts]))
    except Exception as e: r["METEOR"] = f"unavailable ({type(e).__name__}; run nltk.download('wordnet'))"
    try:
        from bert_score import score
        r["BERTScore-F1"] = float(score([res[i][0] for i in gts], [gts[i][0] for i in gts], lang="en")[2].mean())
    except Exception: r["BERTScore-F1"] = "unavailable (pip install bert-score)"
    return r


def image_level(P, df, thr):
    r = {}
    if len(P["dice"]): r["Dice"] = float(P["dice"].mean()); r["Dice_ci"] = ci(P["dice"], np.mean)
    if len(P["prob"]):
        y = np.array([x["pathology"] == "malignant" for x in df.att]).astype(int)
        if 0 < y.sum() < len(y):
            r["AUROC"] = float(roc_auc_score(y, P["prob"]))
            idx = np.arange(len(y)); r["AUROC_ci"] = ci(idx, lambda i: roc_auc_score(y[i], P["prob"][i]) if 0 < y[i].sum() < len(i) else np.nan)
            pred = P["prob"] >= thr
            r["sensitivity"], r["specificity"] = float((pred & (y == 1)).sum() / y.sum()), float((~pred & (y == 0)).sum() / (y == 0).sum())
        for k in ("birads", "pathology", "outline", "size", "orientation"):
            if k in schema:
                gt = np.array([schema[k].index(x[k]) if x.get(k) in schema[k] else -1 for x in df.att])
                r[f"{k}_acc"] = float((np.array(P["attr"][k]) == gt).mean())
    return r


def youden(P, df):
    if not len(P["prob"]): return 0.5
    y = np.array([x["pathology"] == "malignant" for x in df.att]).astype(int)
    ts = np.unique(P["prob"]); j = [((P["prob"] >= t) & (y == 1)).sum() / max(y.sum(), 1) + ((P["prob"] < t) & (y == 0)).sum() / max((y == 0).sum(), 1) for t in ts]
    return float(ts[int(np.argmax(j))])


sets = {"val": pd.read_pickle(run / "split_val.pkl"), "test": pd.read_pickle(run / "split_test.pkl")}
if a.ext:
    sets["external"] = D.load_dataset(a.ext, a.ext_root, expert=False)
    print(f"EXTERNAL set (evaluation only): {len(sets['external'])} samples")
preds = {k: predict(v) for k, v in sets.items()}
thr = youden(preds["val"], sets["val"]); print(f"decision threshold from validation: {thr:.3f}")
res = {}
for k, df in sets.items():
    if k == "val": continue
    c, per = clinical(preds[k]["texts"], df)
    res[k] = {"n": len(df), **image_level(preds[k], df, thr), "NLP": nlp(preds[k]["texts"], df), "clinical": c}
    if df.device.nunique() > 1:   # scanner-wise factual error rate
        res[k]["factual_error_by_device"] = {d: float(per[(df.device == d).values, 1].sum() / max(per[(df.device == d).values, 0].sum(), 1)) for d in df.device.unique()}
    pd.DataFrame({"generated": preds[k]["texts"], "reference": [D.build_report(x) for x in df.att]}).to_csv(run / f"reports_{k}.csv", index=False)
if "external" in res:  # Delta P = P_internal - P_external
    flat = lambda r: {**{m: v for m, v in r.items() if isinstance(v, float)}, **{m: v for m, v in r["NLP"].items() if isinstance(v, float)},
                      **{m: v for m, v in r["clinical"].items() if isinstance(v, float)}}
    fi, fe = flat(res["test"]), flat(res["external"])
    res["delta_internal_minus_external"] = {m: fi[m] - fe[m] for m in fi if m in fe}
json.dump(res, open(run / "metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2))
