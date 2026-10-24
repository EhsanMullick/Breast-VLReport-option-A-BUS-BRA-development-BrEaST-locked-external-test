# Breast-VLReport (option A): BUS-BRA development, BrEaST locked external test

Files: `data.py` (loaders, patient-level split, template reports, tokenizer) · `model.py` · `train.py` · `evaluate.py` · `run_all.sh`

## Read this first
* **Reports are templates, not radiologist text.** Neither dataset has narrative reports. Each report is generated from
  BI-RADS, pathology and mask-derived geometry (relative size, outline regular/irregular, wider/taller than tall).
  These geometric descriptors are *proxies computed from the mask*, not BI-RADS shape/margin labels; their thresholds
  (`SIZE_T`, `SOLIDITY_T`, `CIRC_T` in `data.py`) are arbitrary. BLEU/CIDEr measure template fidelity only.
* **"Factual error rate" / "hallucination rate"** compare facts stated in the generated report with these structured
  labels. Read them together with `omission_rate`: a model that states nothing has 0 errors. They do not replace
  blinded radiologist review (proposal sec. 10).
* **Not tested on real data.** I ran the whole pipeline only on synthetic images (CPU, random weights). Column/file-name
  matching for BUS-BRA and BrEaST follows my understanding of their layouts; the loaders print the detected columns and
  assert on mismatch, so check that output on the first run and adjust `_find()` keys if needed.
* Decision threshold (sens/spec) comes from the dev-site validation split; the external set is evaluate-only.

## Not implemented
Multi-view fusion (single image per sample), clinical-consistency loss (ablation F), CLIP+GPT and BLIP-2 baselines,
attention-rollout/explainability maps, paired tests (McNemar/DeLong/Wilcoxon) and radiologist-rating analysis.
Baselines have no segmentation/attribute heads, so Dice/AUROC are reported only for the lesion-aware variants.
BioGPT is not wired in (HF has no cross-attention for it); `--decoder gpt2 --hf_tok gpt2` gives a pretrained GPT-2 decoder.

## Run
```
pip install -r requirements.txt
python train.py --dev busbra --dev_root /path/BUSBRA --out runs/vl
python evaluate.py --run runs/vl --ext breast --ext_root /path/BrEaST
```
Outputs: `runs/<name>/metrics.json` (incl. bootstrap CIs, per-scanner error, ΔP = internal − external), `reports_*.csv`, `split_*.pkl`.
