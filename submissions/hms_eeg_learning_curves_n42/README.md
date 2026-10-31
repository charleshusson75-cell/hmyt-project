# How Many Labelled EEG Windows Does Each Model Family Need? A Learning-Curve Study of Harmful Brain Activity Classification

- **Topic:** self-proposed, approved by the instructor by email on 21 September 2026. It is no. 12 in the class
  project list, under its old working title "ECG DNN adapted to EEG study". The study no longer adapts an ECG
  network, so the title changed.
- **Team:** N42, one member (student ID 20260147S)
- **Main models:** planned: mean-vote baseline, LightGBM, a tabular foundation model, a 1D CNN
- **Key result:** none at this stage (proposal)

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

- `report/`: Lab 1 report (`N42_Lab1_Report.docx`), literature matrix (`literature_matrix.md`, also as `.xlsx`), `references.bib`
- `slides/`: weekly reports
- `docs/ai_disclosure_log.md`: how AI tools were used

## Data

HMS Harmful Brain Activity Classification (Kaggle): https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification.
De-identified competition data. It is not copied here because the competition rules do not allow redistribution:
sign in to Kaggle and accept the rules to get it. 106,800 labelled windows, 17,089 recordings, 1,950 patients.

## How to run (reproducibility)

Nothing to run at this stage: `gd2-dataset-eda` adds the EDA notebook.
