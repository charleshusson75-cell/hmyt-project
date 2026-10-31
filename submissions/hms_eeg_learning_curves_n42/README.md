# How Many Labelled EEG Windows Does Each Model Family Need? A Learning-Curve Study of Harmful Brain Activity Classification

- **Topic:** self-proposed, approved by the instructor by email on 21 September 2026. It is no. 12 in the class
  project list, under its old working title "ECG DNN adapted to EEG study". The study no longer adapts an ECG
  network, so the title changed.
- **Team:** N42, one member (student ID 20260147S)
- **Main models:** mean-vote baseline and LightGBM (baselines), TabPFN, MiniRocket, 1D CNN
- **Key result:** test KL on windows with ≥ 10 votes, all 14,215 training recordings: LightGBM 0.813, MiniRocket 0.641, 1D CNN 0.638 (mean-vote baseline 0.975); read `report/M2_preprocessing_and_baselines.md` for how solid this is

## Question

How does each family of models improve as the number of labelled EEG recordings grows? One frozen split by patient,
the same training recordings for every model, N = 100, 300, 1,000, 3,000, 10,000 and all, 3 seeds.

## Branches

| Branch | Milestone | What it adds |
|---|---|---|
| `gd1-proposal` | M1 | Problem, costs of errors, literature matrix (Lab 1 report) |
| `gd2-dataset-eda` | M1 | EDA notebook and its figures |
| `gd3-preprocessing` | M2 | Protocol v2 preprocessing, patient-disjoint split, leakage audit |
| `gd4-baseline-model` | M2 | Baselines, first model comparison, results of all 87 runs |

Each branch starts from the previous one, so `gd4-baseline-model` holds everything. As the course's AI guide
asks, milestones are reviewed here in the fork; the only Pull Request comes with the final paper.

Everything was put in this repository on 31 October 2026, when I set up the GitHub process for this midterm
submission, so all commits carry that date. The work itself was done earlier: Lab 1 went in on Teams in mid-October and
the weekly reports went on the class Drive (copies of weeks 4 to 7 in `slides/`).

## Folder

- `report/`: Lab 1 report (`N42_Lab1_Report.docx`), literature matrix (`literature_matrix.md`, also as `.xlsx`), `references.bib`, Milestone 2 report (`M2_preprocessing_and_baselines.md`)
- `src/eda/`: EDA notebook and the data module it used (with outputs on Kaggle: https://www.kaggle.com/code/charleshusson/hms-eda-ipynb)
- `src/`: data preparation and split, scoring, one-run script, models, curve notebook, analysis scripts (`src/README.md`)
- `results/`: all 87 runs, tables and figures (`results/README.md`)
- `docs/reproducibility_checklist.md`: the course checklist, filled in
- `slides/`: weekly reports
- `docs/ai_disclosure_log.md`: how AI tools were used

## Data

HMS Harmful Brain Activity Classification (Kaggle): https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification.
De-identified competition data. It is not copied here because the competition rules do not allow redistribution:
sign in to Kaggle and accept the rules to get it. 106,800 labelled windows, 17,089 recordings, 1,950 patients.

## How to run (reproducibility)

```bash
pip install -r requirements.txt
python src/prepare.py --comp <folder of the HMS competition> --out prepared
```

`prepare.py` builds the windows and the split once. It stops if a patient or a recording falls in two splits.

```bash
python src/run.py --model lgbm --n 1000 --seed 0 --data prepared --out new_results
```

One run = one model, one N, one seed; the model names are the files in `src/models/`. The full curve is
`src/notebooks/hms_curve.ipynb` on a Kaggle GPU (attach the competition and a dataset holding `src/`, Internet on):
every model, size and seed, then the tables and figures. Seeds are fixed, and the test set is the same in all 87 runs.
