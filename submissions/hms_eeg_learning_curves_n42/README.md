# How Many Labelled EEG Windows Does Each Model Family Need? A Learning-Curve Study of Harmful Brain Activity Classification

- **Topic:** self-proposed, approved by the instructor by email on 21 September 2026. It is no. 12 in the class
  project list, under its old working title "ECG DNN adapted to EEG study". The study no longer adapts an ECG
  network, so the title changed.
- **Team:** N42, one member (student ID 20260147S)
- **Final paper:** `report/N42_final_paper.pdf` (also `.docx`); **slides:** `slides/final_report_N42.pptx`
- **Models:** mean-vote baseline; logistic regression (LR) and LightGBM on 160 band-power features; TabPFN (pretrained,
  same features); MiniRocket; LR on MiniRocket + band power; 1D CNN (xresnet1d-18), with and without fine-tuning on
  the windows with ≥ 10 votes
- **Key result** (test KL on the 16,255 windows with ≥ 10 votes of 1,053 patients, 10-fold cross-validation; lower is
  better): with 100 training recordings the LR on band power is best (0.906; baseline 1.054); from 300 to 3,000 the
  LR on MiniRocket + band power leads without a significant margin; from 10,000 the fine-tuned CNN is clearly best
  (0.566 vs 0.685, paired difference −0.119, 95% CI −0.161 to −0.080), but without fine-tuning it only ties the best
  feature model.

## Question

How many labelled recordings does each model family need? Every model is trained on the same nested subsets of
training recordings (N = 100, 300, 1,000, 3,000, 10,000 and all, about 13,700) and scored on the same patients.
Patients are split once into 10 folds; each fold is the test set in turn, so every patient is tested once.

## Branches

| Branch | Milestone | What it adds |
|---|---|---|
| `gd1-proposal` | M1 | Problem, costs of errors, literature matrix (Lab 1 report) |
| `gd2-dataset-eda` | M1 | EDA notebook and its figures |
| `gd3-preprocessing` | M2 | Protocol v2 preprocessing, patient-disjoint split, leakage audit |
| `gd4-baseline-model` | M2 | Baselines, first model comparison, results of all 87 runs |
| `gd5-proposed-model` | M3 | Progress test 2; training recipe 2, CNN + fine-tuning, same-learner LR models (ablation of the input), results of the recipe-2 run |
| `gd6-evaluation-xai` | M4 | 10-fold cross-validation, clinical metrics with intervals, calibration, error analysis, explainability; final results |
| `gd7-ieee-paper` | Final | IEEE-style paper, final slides, final README and checklist |
| `submit/hms_eeg_learning_curves-n42` | Final | Same content as `gd7-ieee-paper`: the branch of the Pull Request |

Each branch starts from the previous one, so `gd7-ieee-paper` holds everything. As the course's AI guide asks,
milestones were reviewed in the fork and the only Pull Request comes with the final paper.

The repository was set up on 31 October 2026 for the midterm, so the commits of `gd1`–`gd4` carry that date although the
work was done earlier (Lab 1 on Teams in mid-October, weekly reports on the class Drive, copies in `slides/`). The M3 and
M4 code and results were committed on 19 November 2026 with the final paper; the code is the version that produced the
committed results (Kaggle runs of 17–19 November).

## Folder

- `report/`: final paper (`N42_final_paper.pdf`, `.docx`), Lab 1 report, literature matrix (`.md`, `.xlsx`),
  `references.bib`, Milestone 2 report, progress test 2 (`kiem_tra_lan_2.md`)
- `src/`: data preparation and split, scoring, metrics, explainability, run scripts, models, curve notebook, tests,
  analysis scripts (`src/README.md`)
- `results/`: `final_cv/` (the final results: 560 runs over 10 folds, tables, figures and the scripts behind every number
  of the paper), `m3_recipe2/` (recipe-2 run, 17 November), and at the top level the 87 runs of the Milestone 2 report
  (`results/README.md`)
- `slides/`: weekly reports (weeks 4–7) and the final presentation
- `docs/reproducibility_checklist.md`: the course checklist, filled in; `docs/ai_disclosure_log.md`: how AI tools were used

## Data

HMS Harmful Brain Activity Classification (Kaggle): https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification.
De-identified competition data. It is not copied here because the competition rules do not allow redistribution:
sign in to Kaggle and accept the rules to get it. 106,800 labelled windows, 17,089 recordings, 1,950 patients.

## How to run (reproducibility)

```bash
pip install -r requirements.txt
python src/prepare.py --comp <folder of the HMS competition> --out prepared
```

`prepare.py` builds the windows and the 10 patient folds once. It stops if a patient or a recording falls in two splits.

```bash
python src/run.py --model lgbm --n 1000 --seed 0 --data prepared --out new_results
python src/run.py --model lgbm --n 1000 --seed 0 --fold 3 --data prepared --out new_results/cv/fold_3
```

One run = one model, one N, one seed, one test fold (`--fold`, 0 by default); the model names are the files in
`src/models/`. The full study is `src/notebooks/hms_curve.ipynb` on a Kaggle GPU (attach the competition and a dataset
holding `src/`, Internet on): fold 0 with 3 seeds, folds 1–9 with seed 0 (`FOLDS`, over two sessions), then the pooled
tables, the clinical metrics and, with `RUN_XAI = True`, the explanations (`src/xai_run.py`). Seeds are fixed and the
CNN runs deterministically: 69 runs repeated three weeks apart gave identical predictions.

Every number and figure of the paper, from the result files:

```bash
python results/final_cv/analysis/report_numbers.py results/final_cv src
python results/final_cv/analysis/report_figures.py results/final_cv src
```

They need the prediction files (`preds/`, written by `run.py`), which are not committed (size); the tables they wrote are
committed next to them.

## Known limitations

One dataset (no external validation); 10 s windows while the experts saw 50 s and spectrograms; seed variation and the
CNN learning rate from fold 0 only; fine-tuning given to the CNN only; 4 of 564 runs (all CNN) failed in every session and
are missing (`results/final_cv/README.md`); TabPFN was not run with all training recordings (it is pretrained for at most
10,000 training samples, `src/models/tabpfn.py`, `MAX_N`), so its curve stops at N = 10,000 and it is not in the
Milestone 4 table (`m4_common.csv`, all recordings). Section V-D of the paper lists them all.

## Which numbers come from which table

A model's KL at a training size (`results/final_cv/analysis/kl_table.csv`, paper Table II) uses every fold where that
model ran. Paired differences use only the folds where both models ran, and `best_by_N.csv` ranks the models on the
windows that every model has at that size (`kl_table_common.csv`). The two differ only where a run failed: at N = 3,000
the LR on MiniRocket + band power scores 0.732 over 10 folds (`kl_table.csv`) and 0.744 over the 9 folds it shares with
the fine-tuned CNN (`best_by_N.csv`).

## Preprocessing: Lab 1 vs final results

Every final result uses protocol v2 (`src/hms_data.py`, `PREP_VERSION = 2`): 200 Hz, 60 Hz notch, 0.3–75 Hz band-pass.
Lab 1 (October) used the first protocol, 100 Hz and 0.5–45 Hz, which was replaced on 23 October 2026.
`src/eda/hms_data.py` is that first-protocol module, kept only because the Lab 1 EDA notebook (`src/eda/hms_eda.ipynb`)
imports it; no result of the final paper uses it.
