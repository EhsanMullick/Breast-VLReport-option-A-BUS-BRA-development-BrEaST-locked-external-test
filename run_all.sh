#!/usr/bin/env bash
# Option A: develop on BUS-BRA (patient-level 10-fold, fold 0), lock BrEaST as the external test.
set -e
BUSBRA=/path/to/BUSBRA          # unzip of https://zenodo.org/record/7730708
BREAST=/path/to/BrEaST          # BrEaST dataset (CC-BY 4.0), contains the .xlsx + images/masks
for M in cnn_lstm cnn_transformer vit_transformer swin_transformer breast_vlreport; do
  python train.py --dev busbra --dev_root $BUSBRA --model $M --out runs/$M
  python evaluate.py --run runs/$M --ext breast --ext_root $BREAST
done
# Ablations (proposal sec. 14): G = direct image->report, H = +lesion, I = +attributes, D = no contrastive
python train.py --dev busbra --dev_root $BUSBRA --model breast_vlreport --no_lesion       --out runs/abl_G
python train.py --dev busbra --dev_root $BUSBRA --model breast_vlreport --no_attr         --out runs/abl_H
python train.py --dev busbra --dev_root $BUSBRA --model breast_vlreport --no_contrastive  --out runs/abl_D
# Option B (secondary): BrEaST as dev with expert descriptors, BUS-BRA as external
# python train.py --dev breast --dev_root $BREAST --expert_attrs --folds 5 --bs 8 --out runs/optB
# python evaluate.py --run runs/optB --ext busbra --ext_root $BUSBRA
