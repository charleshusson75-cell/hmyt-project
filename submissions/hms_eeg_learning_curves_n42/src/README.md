# Code (protocol v2)

One model, one training size, one seed per run. A learning curve is a loop over runs.

- `hms_data.py`: `prepare()` parquet to arrays and split (once), `load()`, `fold_split(meta, k)` (fold k = test, k+1 =
  validation), `make_split(data, N, seed, test_fold=k)`
- `hms_eval.py`: scores, bootstrap over patients, tables and figures; `cv_*`: results pooled over the folds
- `hms_metrics.py`: clinical metrics (AUC, PR-AUC, sensitivity, specificity, F1, Brier), calibration on vote shares,
  patient bootstrap, error analysis
- `hms_xai.py`, `xai_run.py`: LightGBM SHAP, LR coefficients, CNN channel occlusion and Grad-CAM (correct and
  misclassified windows), run as two processes
- `prepare.py`, `run.py`: command-line entry points (`run.py --fold k` for cross-validation)
- `models/`: `floor` (mean-vote baseline), `logreg_band`, `lgbm`, `tabpfn`, `minirocket`, `logreg_both`, `cnn1d`,
  `cnn1d_ft` (fine-tuning on >= 10-vote windows), `deep` (recipe shared by deep models)
- `notebooks/hms_curve.ipynb`: every model, size, seed and fold on a Kaggle GPU, then all tables, metrics and explanations
- `tests/test_minirocket.py`: MiniRocket transform against aeon, PyTorch head against scikit-learn
- `analysis/`: robustness and label analyses on a finished run. They need `results/preds/`, which `run.py` writes and which
  is not committed, and `HMS_TRAIN_CSV` set to the competition's `train.csv`.
- `eda/`: the Lab 1 data analysis only, with the first-protocol data module it imports (`eda/hms_data.py`: 100 Hz,
  0.5-45 Hz). Not used by any final result: those use `hms_data.py` above (200 Hz, 0.3-75 Hz)
