# Training code for the S6 all-data model and the CoAtNet 0.949 / 0.945 models

Added for a leakage-controlled retraining request (discussion 750180). Every file is byte-identical to what ran;
`SHA256SUMS` lists them.

- `s6/train_s6f0_all.py` (sha256 ae18d2e1…): Heliosli's `compact_train.py` (public notebook, Apache-2.0) + patches P1–P8
  (`train_s6f0.py`: on-the-fly DICOM loading with precomputed SKM-TEA anchors, frozen `--pool`, split guards, atomic
  resume) + A1–A4 (`--all-data`). `compact_train_to_train_s6f0_all.diff` shows every change. The production argv is in
  `run_all_alldata.sh`; `inputs.sha256` pins every training input. Epoch-16 weights, no SWA/EMA, nothing selected on Gold-58.
  Heliosli's `cloud_code` and `anatomy.pt` are NOT copied: attach heliosli/rsna-orthofoundation-fold0-epoch7-20261004
  (`heliosli.SHA256SUMS` / `cloud_code.SHA256SUMS` pin the files used).
- `s6/make_data.py` → `data/targets_s6f0.csv` (every non-gold `L|prior` = labels_v1_blendjev; `L|strict` blank; arm E_prior =
  unit weights, weight-normalised BCE; gold rows blank and dropped). `s6/freeze_pool.py` → `data/pool_ok.json` (4,321 =
  4,263 training + 58 gold; 0 precompute failures; `pool_ok.counts.json`). The 4,263 = folds_v0 folds 0–4 (4,349) minus 86
  studies whose DICOMs we had deleted locally on 2026-09-29 to free disk space (they are still in the competition data;
  this was our storage, not a host change, and not a quality filter). `pool_ok.json` is the authoritative membership.
- `labels/labels_v1_blendjev.parquet` + `build_blendjev.py` (rebuilds it exactly from labels_v1_blend in
  nartaa/rsna-knee-hpo-assets and riadmohamed42/jev-knee-labels, CC0).
- `coatnet/`: `train_knee.py`, `ext_trainer.py` and `RUNS.md` (argv, OAI masking, Gold-ranked SWA epochs, NDA route).

No images, reports or OAI subject data are included.

## Licence (code)

Per the competition's public code-sharing rule, the code in `training_code/` is shared under the Apache License 2.0
(OSI-approved, no limit on commercial use). Our patches and scripts are Apache-2.0; Heliosli's `compact_train.py` and
dreaddevelopment's Raptor training notebook, on which `train_s6f0*.py` and `train_knee.py` are based, are public Kaggle
notebooks under Apache-2.0. The JEV labels are CC0. Folds: `folds_v0.json` in nartaa/rsna-knee-hpo-assets (sha256 12bebc89…).
