"""Datasets, structured-attribute extraction, template reports, tokenizer, patient-level splits.

IMPORTANT: neither BUS-BRA nor BrEaST ships narrative radiology reports. Reports here are TEMPLATE
reports built from structured annotations (BI-RADS, pathology, mask-derived geometry and, for BrEaST,
expert BI-RADS descriptors). Text metrics therefore measure fidelity to the template, not radiologist style.
"""
import re, random, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

IMG = 224
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

# ---- attribute <-> report slot definitions -------------------------------------------------------------
SLOT_KEY = {  # attribute name -> phrase used in the report ("<key> is <phrase>.")
    "size": "relative size", "outline": "outline", "orientation": "orientation", "shape": "shape",
    "margin": "margin", "echogenicity": "echogenicity", "posterior": "posterior feature",
    "pathology": "assessment", "birads": "birads category"}
PHRASE = {  # value -> phrase (kept distinct from each other inside one attribute)
    "pathology": {"benign": "most consistent with a benign lesion", "malignant": "suspicious for malignancy"},
    "orientation": {"wide": "wider than tall", "tall": "taller than wide"},
}
ORDER_FIND = ["size", "outline", "orientation", "shape", "margin", "echogenicity", "posterior"]
ORDER_IMP = ["pathology", "birads"]
# mask-derived geometric proxies (thresholds are arbitrary constants -> document/tune them)
SIZE_T = (0.02, 0.08)           # fraction of image area: small < 0.02 <= medium < 0.08 <= large
SOLIDITY_T, CIRC_T = 0.90, 0.60  # outline "irregular" if solidity<0.90 or circularity<0.60


def phrase(attr, v):
    return PHRASE.get(attr, {}).get(v, v)


def build_report(attrs: dict) -> str:
    f = [f"{SLOT_KEY[a]} is {phrase(a, attrs[a])}." for a in ORDER_FIND if attrs.get(a)]
    i = [f"{SLOT_KEY[a]} is {phrase(a, attrs[a])}." for a in ORDER_IMP if attrs.get(a)]
    return "findings: a lesion is seen. " + " ".join(f) + " impression: " + " ".join(i)


def norm_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+([.,;:])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_report(text: str, attr_names) -> dict:
    """Read stated facts back out of a (generated) report. Returns {attr: value | '<invalid>'}."""
    text, out = norm_text(text), {}
    for a in attr_names:
        m = re.search(rf"{re.escape(SLOT_KEY[a])} is ([^.]+)\.", text)
        if m:
            inv = {phrase(a, v): v for v in ALL_VALUES.get(a, [])}
            out[a] = inv.get(m.group(1).strip(), "<invalid>")
    return out


ALL_VALUES = {}  # filled by register_values(); used by the parser to map phrases back to values
def register_values(schema):
    for a, vals in schema.items():
        ALL_VALUES[a] = list(vals)


# ---- mask geometry -------------------------------------------------------------------------------------
def mask_attrs(mask: np.ndarray) -> dict:
    from skimage.measure import label, regionprops
    lab = label(mask > 0)
    if lab.max() == 0:
        return {}
    r = max(regionprops(lab), key=lambda r: r.area)
    frac = r.area / mask.size
    circ = 4 * np.pi * r.area / (r.perimeter ** 2 + 1e-6)
    h, w = r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1]
    return {"size": "small" if frac < SIZE_T[0] else "medium" if frac < SIZE_T[1] else "large",
            "outline": "irregular" if (r.solidity < SOLIDITY_T or circ < CIRC_T) else "regular",
            "orientation": "wide" if w >= h else "tall"}


def union_mask(paths):
    m = None
    for p in paths:
        a = np.array(Image.open(p).convert("L")) > 0
        m = a if m is None else (m | a)
    return m.astype(np.uint8)


# ---- dataset adapters ----------------------------------------------------------------------------------
def _find(df, *keys):
    cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for k in keys:
        for n, c in cols.items():
            if k in n:
                return c
    return None


def load_busbra(root) -> pd.DataFrame:
    root = Path(root)
    csv = next(iter(root.rglob("bus_data.csv")), None) or next(iter(root.rglob("*.csv")))
    df = pd.read_csv(csv)
    print("[BUS-BRA] columns:", list(df.columns))
    c_id, c_bi, c_pa = _find(df, "id"), _find(df, "birads"), _find(df, "pathology")
    c_case, c_dev = _find(df, "case"), _find(df, "device", "scanner")
    imgs = {p.stem: p for p in root.rglob("*.png")}
    rows = []
    for _, r in df.iterrows():
        sid = str(r[c_id])
        ip = imgs.get(sid)
        mp = imgs.get(sid.replace("bus_", "mask_")) or imgs.get("mask_" + sid.split("_", 1)[-1])
        if ip is None or mp is None:
            continue
        m = np.array(Image.open(mp).convert("L"))
        a = mask_attrs(m)
        if not a:
            continue
        a["birads"] = str(r[c_bi])[0]
        a["pathology"] = str(r[c_pa]).strip().lower()
        grp = str(r[c_case]) if c_case else re.sub(r"\D", "", sid.split("_")[-1].split("-")[0])
        rows.append(dict(sample=sid, img=str(ip), mask=str(mp), group=f"busbra_{grp}",
                         device=str(r[c_dev]) if c_dev else "NA", att=a, source="busbra"))
    assert rows, "No BUS-BRA samples matched. Check file names vs. CSV IDs."
    return pd.DataFrame(rows)


def load_breast(root, expert=False) -> pd.DataFrame:
    root = Path(root)
    xl = next(iter(root.rglob("*.xlsx")))
    df = pd.read_excel(xl)
    print("[BrEaST] columns:", list(df.columns))
    c_f, c_bi, c_cl = _find(df, "filename", "image"), _find(df, "birads"), _find(df, "classification")
    c_id = _find(df, "caseid", "case")
    ex = {"shape": _find(df, "shape"), "margin": _find(df, "margin"),
          "echogenicity": _find(df, "echogenicity"), "posterior": _find(df, "posterior")}
    files = {p.name: p for p in root.rglob("*.png")}
    rows = []
    for _, r in df.iterrows():
        cls = str(r[c_cl]).lower()
        if "normal" in cls or not (("benign" in cls) or ("malignant" in cls)):
            continue
        fn = str(r[c_f])
        ip = files.get(fn)
        stem = Path(fn).stem
        mps = [p for n, p in files.items() if n.startswith(stem + "_tumor")]
        if ip is None or not mps:
            continue
        a = mask_attrs(union_mask(mps))
        if not a:
            continue
        a["birads"] = re.sub(r"\D", "", str(r[c_bi]))[:1] or "NA"
        a["pathology"] = "malignant" if "malignant" in cls else "benign"
        if expert:
            for k, c in ex.items():
                v = str(r[c]).strip().lower() if c else ""
                if v and v != "nan":
                    a[k] = "not circumscribed" if (k == "margin" and not v.startswith("circ")) else v
        rows.append(dict(sample=stem, img=str(ip), mask=str(mps[0]), mask_all=[str(p) for p in mps],
                         group=f"breast_{r[c_id] if c_id else stem}", device="BrEaST", att=a, source="breast"))
    assert rows, "No BrEaST samples matched. Print the columns above and adjust _find() keys."
    return pd.DataFrame(rows)


def load_dataset(name, root, expert=False):
    df = load_busbra(root) if name == "busbra" else load_breast(root, expert)
    return df[df.att.map(lambda a: a.get("birads", "NA") in "2345")].reset_index(drop=True)


def make_split(df, folds=10, fold=0, seed=0):
    """Patient-level split: test=fold, val=fold+1, train=rest; stratified by pathology; asserts no leakage."""
    y = df.att.map(lambda a: a["pathology"]).values
    sg = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    parts = [idx for _, idx in sg.split(df, y, df.group)]
    te, va = parts[fold % folds], parts[(fold + 1) % folds]
    tr = np.setdiff1d(np.arange(len(df)), np.concatenate([te, va]))
    g = lambda i: set(df.group.iloc[i])
    assert not (g(tr) & g(va)) and not (g(tr) & g(te)) and not (g(va) & g(te)), "patient leakage!"
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def build_schema(df):
    s = {}
    for a in df.att:
        for k, v in a.items():
            s.setdefault(k, set()).add(v)
    return {k: sorted(v) for k, v in s.items()}


# ---- tokenizer -----------------------------------------------------------------------------------------
class Tok:
    def __init__(self, texts=None, hf=None, max_len=96):
        self.max_len, self.hf = max_len, None
        if hf:
            from transformers import AutoTokenizer
            self.hf = AutoTokenizer.from_pretrained(hf)
            self.pad = self.eos = self.bos = self.hf.eos_token_id
            self.V = len(self.hf)
        else:
            self.pad, self.bos, self.eos, self.unk = 0, 1, 2, 3
            words = sorted({w for t in texts for w in self.split(t)})
            self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + words
            self.stoi = {w: i for i, w in enumerate(self.itos)}
            self.V = len(self.itos)

    @staticmethod
    def split(t):
        return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*|[.,;:]", t.lower())

    def encode(self, t):
        ids = self.hf.encode(norm_text(t)) if self.hf else [self.stoi.get(w, self.unk) for w in self.split(t)]
        ids = [self.bos] + ids[: self.max_len - 2] + [self.eos]
        return ids + [self.pad] * (self.max_len - len(ids))

    def decode(self, ids):
        ids = [int(i) for i in ids]
        if self.hf:
            cut = ids.index(self.eos, 1) if self.eos in ids[1:] else len(ids)
            return norm_text(self.hf.decode(ids[1:cut]))
        out = []
        for i in ids:
            if i == self.eos: break
            if i > 3: out.append(self.itos[i])
        return norm_text(" ".join(out))


# ---- torch dataset -------------------------------------------------------------------------------------
def letterbox(im, resample):
    w, h = im.size
    s = max(w, h)
    bg = Image.new(im.mode, (s, s), 0)
    bg.paste(im, ((s - w) // 2, (s - h) // 2))
    return bg.resize((IMG, IMG), resample)


class BUSData(Dataset):
    def __init__(self, df, schema, tok, train=False):
        self.df, self.schema, self.tok, self.train = df, schema, tok, train
        self.reports = [build_report(a) for a in df.att]

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = letterbox(Image.open(r["img"]).convert("RGB"), Image.BILINEAR)
        mk = letterbox(Image.fromarray(union_mask(r.mask_all if isinstance(r.get("mask_all"), list) else [r["mask"]]) * 255),
                       Image.NEAREST)
        x = np.asarray(im, dtype=np.float32) / 255.0
        m = (np.asarray(mk) > 127).astype(np.float32)
        if self.train:
            if random.random() < 0.5: x, m = x[:, ::-1], m[:, ::-1]
            x = np.clip((x - 0.5) * random.uniform(0.85, 1.15) + 0.5 + random.uniform(-0.08, 0.08), 0, 1)
        x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy()).float()
        lab = {a: (self.schema[a].index(r.att[a]) if a in r.att and r.att[a] in self.schema[a] else -100)
               for a in self.schema}
        return dict(img=x, mask=torch.from_numpy(m.copy())[None], ids=torch.tensor(self.tok.encode(self.reports[i])),
                    labels=torch.tensor([lab[a] for a in self.schema]), idx=i)
