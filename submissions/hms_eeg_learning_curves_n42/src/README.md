# Code (protocol v2)

One model, one training size, one seed per run. A learning curve is a loop over runs.

- `hms_data.py`: `prepare()` parquet to arrays and split (once), `load()`, `make_split(data, N, seed)`
- `hms_eval.py`: scores, bootstrap over patients, tables and figures
- `prepare.py`, `run.py`: command-line entry points
- `models/`: `floor` (mean-vote baseline), `lgbm`, `tabpfn`, `minirocket`, `cnn1d`, `deep` (recipe shared by deep models)
- `notebooks/hms_curve.ipynb`: every model, size and seed on a Kaggle GPU, then all tables and figures
- `tests/test_minirocket.py`: MiniRocket transform against aeon, PyTorch head against scikit-learn
- `analysis/`: robustness and label analyses on a finished run. They need `results/preds/`, which `run.py` writes and which
  is not committed, and `HMS_TRAIN_CSV` set to the competition's `train.csv`.
- `eda/`: the Lab 1 data analysis, with the first-protocol data module it used
