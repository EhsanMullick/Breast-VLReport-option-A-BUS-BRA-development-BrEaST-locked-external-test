"""Staged training. Proposed model: S1 visual pretraining (seg+attrs) -> S2 image-text alignment (contrastive)
-> S3 report CE -> S4 joint. Baselines: single CE stage.  NOTE: the clinical-consistency loss (ablation F) is NOT implemented."""
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
import data as D
from model import ReportModel, compute_losses

p = argparse.ArgumentParser()
p.add_argument("--dev", default="busbra", choices=["busbra", "breast"]); p.add_argument("--dev_root", required=True)
p.add_argument("--expert_attrs", action="store_true", help="use BrEaST expert descriptors (option B only)")
p.add_argument("--fold", type=int, default=0); p.add_argument("--folds", type=int, default=10)
p.add_argument("--model", default="breast_vlreport", choices=["breast_vlreport", "cnn_lstm", "cnn_transformer", "vit_transformer", "swin_transformer"])
p.add_argument("--no_lesion", action="store_true"); p.add_argument("--no_attr", action="store_true"); p.add_argument("--no_contrastive", action="store_true")
p.add_argument("--decoder", default=None, choices=["tfm", "gpt2", "lstm"]); p.add_argument("--hf_tok", default=None)
p.add_argument("--epochs", type=int, nargs=4, default=[15, 5, 15, 15], help="S1 S2 S3 S4 (baselines use S3 only)")
p.add_argument("--bs", type=int, default=16); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--enc_lr", type=float, default=5e-5)
p.add_argument("--no_pretrained", action="store_true"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=2); p.add_argument("--limit", type=int, default=0)
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

df = D.load_dataset(a.dev, a.dev_root, a.expert_attrs)
if a.limit: df = df.sample(a.limit, random_state=0).reset_index(drop=True)
tr, va, te = D.make_split(df, a.folds, a.fold, a.seed)
for n, s in zip(["train", "val", "test"], [tr, va, te]):
    s.to_pickle(out / f"split_{n}.pkl")
schema = D.build_schema(tr); D.register_values(schema)
tok = D.Tok(texts=[D.build_report(x) for x in tr.att], hf=a.hf_tok)
print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)} patients-disjoint | schema: {schema}")

enc = {"breast_vlreport": "swin", "swin_transformer": "swin", "vit_transformer": "vit"}.get(a.model, "cnn")
full = a.model == "breast_vlreport"
dec = a.decoder or ("lstm" if a.model == "cnn_lstm" else "tfm")
model = ReportModel(tok, schema, enc, dec, lesion=full and not a.no_lesion,
                    attr=full and not a.no_lesion and not a.no_attr,
                    contrastive=full and not a.no_contrastive, pretrained=not a.no_pretrained).to(dev)

mk = lambda s, t: DataLoader(D.BUSData(s, schema, tok, t), a.bs, shuffle=t, num_workers=a.workers, drop_last=t)
dl_tr, dl_va = mk(tr, True), mk(va, False)
move = lambda b: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

# stage = (name, epochs, loss weights, freeze encoder)
if full:
    W = lambda **k: {"seg": 0, "attr": 0, "con": 0, "ce": 0, **k}
    stages = [("S1_visual", a.epochs[0], W(seg=1, attr=1), False), ("S2_align", a.epochs[1], W(seg=.5, attr=.5, con=1), False),
              ("S3_report", a.epochs[2], W(ce=1), True), ("S4_joint", a.epochs[3], W(ce=1, seg=.3, attr=.3, con=.1), False)]
    stages = [s for s in stages if s[1] > 0]
else:
    stages = [("CE", a.epochs[2], {"ce": 1}, False)]

log = []
for name, ep, w, freeze in stages:
    w = {k: v for k, v in w.items() if v}
    for q in model.enc.parameters(): q.requires_grad = not freeze
    enc_p = [q for q in model.enc.parameters() if q.requires_grad]
    ids_e = {id(q) for q in model.enc.parameters()}
    rest = [q for q in model.parameters() if id(q) not in ids_e and q.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.enc_lr}, {"params": rest, "lr": a.lr}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.enc_lr, a.lr], total_steps=max(1, ep * len(dl_tr)), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best, best_sd = 1e9, None

    def step(b, train):
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            o = model(b); L = compute_losses(model, b, o)
            tot = sum(w[k] * L[k] for k in w if k in L)
        return tot, L

    for e in range(ep):
        model.train(); t0 = time.time()
        for b in dl_tr:
            tot, _ = step(move(b), True)
            opt.zero_grad(); scaler.scale(tot).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        model.eval(); vt, n = 0, 0
        with torch.no_grad():
            for b in dl_va:
                t, _ = step(move(b), False); vt += t.item() * len(b["idx"]); n += len(b["idx"])
        vt /= max(n, 1)
        print(f"[{name}] epoch {e+1}/{ep} train_last={tot.item():.3f} val={vt:.3f} ({time.time()-t0:.0f}s)")
        log.append(dict(stage=name, epoch=e + 1, val=vt))
        if vt < best: best, best_sd = vt, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_sd: model.load_state_dict(best_sd)   # model selection on VALIDATION only

torch.save({"state": model.state_dict(), "args": vars(a), "schema": schema,
            "tok": {"hf": a.hf_tok, "texts": [D.build_report(x) for x in tr.att], "max_len": tok.max_len}}, out / "model.pt")
json.dump(log, open(out / "log.json", "w"))
print("saved", out / "model.pt")
