# Progress Test 2 (Week 10): team N42

| | |
|---|---|
| Team code | `n42` (class team N42) |
| Topic slug | `hms_eeg_learning_curves` |
| Topic | How Many Labelled EEG Windows Does Each Model Family Need? A Learning-Curve Study of Harmful Brain Activity Classification |
| Fork | https://github.com/charleshusson75-cell/hmyt-project |
| Branch | `gd5-proposed-model` |
| Members and roles | 20260147S, single member: project lead and ML architect, data and preprocessing, evaluation, documentation |

All paths below are relative to `submissions/hms_eeg_learning_curves_n42/`. Results come from one Kaggle GPU run of
87 runs (23 October 2026, protocol v2), stored in `results/`.

---

## A. Literature review and research gaps

### A1. Search strategy

| Item | Value |
|---|---|
| Topics searched | The four areas agreed with the instructor on 21 September 2026: EEG classification methods; learning with limited labelled data; foundation models for small data; label disagreement |
| Sources | Candidate papers found with an AI assistant's web search (Claude), then screened by me; every kept paper was opened as a PDF and stored in the team's reference folder |
| Keywords | Not logged; the queries followed the four topics above |
| Years | 2020–2026 |
| Hits / papers kept | Hits were not logged. 12 papers kept: 11 journal papers plus 1 preprint ([4]) that motivated the project and is not counted |
| Inclusion | Peer-reviewed journal paper, Q1 in SCImago, 2020–2026, relevant to one of the four topics |
| Exclusion | arXiv-only papers (except [4], kept as motivation and marked); no other criterion was written down |

Weakness: databases, exact queries and hit counts were not recorded, so the search cannot be reproduced. A logged
search (PubMed and Scopus, fixed queries, hit counts) is planned for the final paper.

### A2. Literature matrix

Numbers are taken from the PDFs; page or section in brackets, as in Table I of the Lab 1 report
(`report/literature_matrix.md`). DOIs were checked on Crossref on 31 October 2026 (`report/references.bib`).

| # | Authors, year | DOI / link | Dataset | Method | Main result (metric) | External validation | Main limitation |
|---|---|---|---|---|---|---|---|
| 1 | Jing et al., 2023 | 10.1212/WNL.0000000000207127 | MGH IIIC EEG: 6,095 EEGs, 2,711 patients, 50,697 segments, 20 experts | CNN (SPaRCNet), 6 classes, multi-rater labels | Share of 20 experts below its ROC: SZ 45%, LPD 20%, GPD 50%, LRDA 75%, GRDA 55%, other 40% (p. e1751, e1755) | No: single centre, held-out patients | Segments, not continuous EEG; no false alarms per hour (p. e1759) |
| 2 | Gemein et al., 2020 | 10.1016/j.neuroimage.2020.117021 | TUH Abnormal v2.0.0, 2,993 recordings | Handcrafted features + RF / SVM / Riemannian vs CNNs and TCN | Accuracy 81–86% for all models; best 86.16% (BD-TCN) (p. 1, Sec. 3.3) | No: held-out test set of the same corpus | Binary task; accuracy may saturate because of label agreement (p. 1) |
| 3 | Banville et al., 2021 | 10.1088/1741-2552/abca18 | PC18 (994 subjects); TUH Abnormal (2,993 recordings) | Self-supervised pretext tasks + linear classifier | Balanced accuracy up to 72.3% (PC18), 79.4% (TUH Abnormal) (Sec. 3.1) | No: splits within each dataset | Split by recording; fixed hyperparameters across data sizes (Sec. 4.4) |
| 4 | Stym-Popper et al., 2026 (preprint, not counted) | arXiv:2608.31013 | Pretraining TUEV, TUAB, PTB-XL, HiRID; 5 test datasets, N = 100 | TabPFN adapted to multichannel time series | Mean AUC 82.5 vs 78.1 for MiniRocket (Table 1) | No: cross-validation within each dataset | Input ≤ 500 values (T × C); not grouped by patient |
| 5 | Hollmann et al., 2025 | 10.1038/s41586-024-08328-6 | 29 classification + 28 regression tabular datasets, ≤ 10,000 samples | Transformer pretrained on synthetic data, in-context (TabPFN) | Normalised ROC AUC 0.939 vs 0.752 for CatBoost (Fig. 4a) | Not applicable: tabular benchmark with random splits | ≤ 10,000 samples and 500 features |
| 6 | Jing et al., 2023 | 10.1212/WNL.0000000000201670 | MGH: 2,711 EEGs, 50,697 segments, 30 experts | Inter-rater reliability study | Pairwise agreement 52% (κ 0.42); agreement with majority 65% (κ 0.61) (p. e1737) | Not applicable: no model | Single centre; some patterns not scored |
| 7 | Sun et al., 2025 | 10.1038/s41746-025-02154-4 | HMS: 1,950 patients, 17,089 records, 106,800 segments; external: 1,532 patients | ImageNet-pretrained vision backbone on EEG mapped to images (VIPEEGNet) | KL 0.225 (5-fold); external KL 0.223 / 0.273 (p. 1, p. 4) | Yes: external cohort of 1,532 patients | Needs the 50 s around the window, so not causal (p. 8) |
| 8 | Tveit et al., 2023 | 10.1001/jamaneurol.2023.1645 | Development 30,493 EEGs; tests of 100, 9,785 and 60 EEGs | CNN (SCORE-AI), normal vs 4 abnormal categories | AUC 0.89–0.96; accuracy 88.3% (p. 805, 809) | Yes: independent test sets | Routine EEG only: ICU patients and neonates excluded (p. 811) |
| 9 | Saab et al., 2020 | 10.1038/s41746-020-0264-0 | Stanford: 136,014 EEGs, 12,430 patients | CNN trained on weak clinical annotations | AUROC 0.93 paediatric, 0.94 adult (p. 1) | No: single institution | Patient separation not reported (p. 3) |
| 10 | Kiessner et al., 2023 | 10.1016/j.nicl.2023.103482 | TUABEX: 15,300 recordings | Same CNNs and TCN as [2], on automatically labelled data | BD-TCN on TUABEXB test: 74.96% trained on TUAB, 83.73% trained on TUABEXB (Table 4) | No: not tested on other datasets (Sec. 4) | Generalisation not verified |
| 11 | Rommel et al., 2022 | 10.1088/1741-2552/aca220 | Sleep Physionet (78 subjects); BCI IV 2a (9 subjects) | 13 EEG data augmentations, 2 CNNs | Up to 45% accuracy gain with few training data (Sec. 4.3) | No: subject-level cross-validation within each dataset | Small non-clinical datasets |
| 12 | Strodthoff et al., 2021 | 10.1109/JBHI.2020.3022989 | PTB-XL: 21,837 ECGs, 18,885 patients | 1D CNNs (xresnet1d101 …), LSTM, wavelet + NN | Macro AUC 0.925 (xresnet1d101) vs 0.849 (wavelet + NN) (Table II) | No: PTB-XL test fold | ECG, not EEG |

### A3. Research gaps

| ID | Gap | Evidence (rows) | How the project addresses it |
|---|---|---|---|
| G1 | No learning curves across model families for these six patterns: the HMS papers train one model on all patients; papers that vary the labelled data do so for other tasks, with accuracy | 1, 7 (one model, all data); 2 (families compared only with all data, binary task); 3, 10, 11 (data size varied, other tasks) | Five families (baseline, LightGBM, TabPFN, MiniRocket, 1D CNN) at N = 100 … 14,215 training recordings, 3 seeds, same split and same training recordings for every model (`results/`) |
| G2 | Results depend on the split, and splits differ; in-context models were tested without patient grouping | 3 (by recording), 4 (stratified CV, no grouping), 8 (patient overlap not reported), 9 (not reported) | One patient-disjoint, class-stratified split, checked by `assert`, frozen for all 87 runs (`src/hms_data.py:145`, `:166`) |
| G3 | Label quality is uneven and tied to the class, but results are rarely split by it | 6 (moderate agreement); 7 (one KL over all windows); 1 (seizure 18.8% of segments with ≥ 3 labels but 1.5% of the ≥ 10-label test set, Table 1) | Primary metric on windows with ≥ 10 votes; KL also reported by number of votes (`results/figures/kl_by_votes_n10000.csv`); the same shift found in HMS (seizure 15% of training votes vs 5% of scored windows, `results/figures/class_mix_train_vs_test.png`) |
| G4 | Pretrained models for small data are rarely compared with strong classical baselines on a patient-level split of clinical EEG | 4, 5 (tabular / time-series foundation models, no clinical EEG patient split); 7 (ImageNet pretraining, no scratch control reported in our matrix) | TabPFN vs LightGBM on the same features (done); ImageNet vs random weights, and CBraMod, planned for M3 (B4) |

### A4. PROBAST+AI appraisal of the most important paper: Jing et al. 2023 (SPaRCNet), row 1

Chosen because HMS is built from its data: the 1,950 patients of HMS match the 1,950 patients of its training set
(Table 1, p. e1752). Codes: Y, PY, PN, N, NI (no information), NA. Page numbers refer to the published PDF.

| Domain | Signalling questions (answer) | Justification (location in the paper) | Risk of bias |
|---|---|---|---|
| 1. Participants and data | Appropriate data source? Y. Appropriate inclusion and exclusion? PN | Retrospective clinical EEGs from one hospital, patients with and without IIIC events (Methods, p. e1751). The model is trained and tested on 10 s segments chosen for expert annotation, not on consecutive EEG; the test set keeps only segments with ≥ 10 labels from the 20 experts (p. e1752). The authors state that false alarms per hour cannot be measured (Discussion, p. e1759) | High |
| 2. Predictors | Defined and assessed the same way for all? Y. Assessed without knowledge of the outcome? Y | The input is the EEG segment itself, the same for every patient (Model Development, p. e1753) | Low |
| 3. Outcome | Outcome defined and determined the same way for all? Y. Determined without knowledge of the model output? Y. Predictors excluded from the outcome definition? NA | The reference is the majority of 19 experts, leaving out the expert being compared (p. e1754). Experts read the same EEG the model sees: for this task the expert reading is the reference standard itself, not incorporation of an extra predictor. Noisy segments were put in the "other" class (p. e1752), so "other" mixes non-IIIC patterns and artefacts | Low |
| 4. Analysis | Enough participants? Y. Split by patient? Y. Calibration assessed? Y. Uncertainty by bootstrap? PY (resampling unit not stated in the main text: NI). Model tuning details? NI (supplement) | Test set of 35,740 segments from 761 patients (Table 1); patients kept in one set, sets balanced by class and agreement (p. e1752–e1753); calibration index compared with experts (p. e1755); bootstrap CIs (p. e1754); no external cohort | Unclear |
| Overall | | Mainly from domain 1: selected segments, single centre | High |

Applicability to our question: low concern for data, classes and labels (same source, same six classes, same ≥ 10-vote
quality threshold). Its metrics (share of experts below the ROC/PR curve, calibration index) differ from ours (KL
divergence), so its numbers cannot be compared directly with ours. Like the paper, our project works on selected
segments, so domain 1 applies to us too (C3).

---

## B. Results and timeline

### B1. Current results

Evaluation set: test windows with ≥ 10 votes, 1,484 windows from 109 patients (patient-disjoint from training).
Metric: KL divergence between predicted and expert vote distributions (lower is better). 95% CI: bootstrap over test
patients, seed 0. Mean over 3 seeds. N = 14,215 training recordings (all).

| Role | Model | Test KL (mean of 3 seeds) | 95% CI (seed 0) | Macro AUC | Source |
|---|---|---|---|---|---|
| Baseline (M2) | Mean-vote baseline | 0.975 | 0.819 – 1.156 | 0.500 | `results/floor_n14215_s*.json` |
| Baseline (M2) | LightGBM, 160 band-power features | 0.813 | 0.631 – 1.054 | 0.747 | `results/lgbm_n14215_s*.json` |
| Compared family | MiniRocket + logistic head | 0.641 | 0.521 – 0.792 | 0.849 | `results/minirocket_n14215_s*.json` |
| Compared family | 1D CNN (xresnet1d-18) | 0.638 | 0.540 – 0.744 | 0.842 | `results/cnn1d_n14215_s*.json` |
| Proposed model (M3) | Not yet: CBraMod fine-tuning and ImageNet-pretrained EfficientNet are designed but not run (B4) | – | – | – | – |

Full learning curves (N = 100 to 14,215): `results/figures/test_kl_hq_table.csv`, `results/figures/learning_curves.png`;
paired differences: `results/figures/paired_pooled.csv`; reading of the results: `report/M2_preprocessing_and_baselines.md`.

Main points, with their limits:
- MiniRocket is significantly better than LightGBM from 3,000 training recordings (paired difference at N = 14,215:
  −0.172, 95% CI −0.283 to −0.061).
- At full size the CNN and MiniRocket tie (−0.003, CI −0.145 to 0.124). The CNN's lead over LightGBM (−0.175) has a
  CI that includes zero (−0.414 to 0.040), and one test patient (155 windows) accounts for all of it.
- Below 1,000 recordings the CNN is the worst model, but this measures a flaw in its stopping rule (B5), not the CNN.

### B2. Ablation (Milestone 3)

Not done yet: there is no proposed model to ablate. The closest existing evidence is not an ablation:

| Configuration | Metric | Difference | Source |
|---|---|---|---|
| Same features, pretrained in-context model (TabPFN) vs trees (LightGBM), N = 300 to 10,000 | Test KL, ≥ 10 votes | TabPFN better by 0.04 to 0.07; every 95% CI includes zero | `results/figures/paired_pooled.csv` |
| Same predictions, test windows weighted per recording, per patient, or without the largest patient (N = 14,215) | Test KL | MiniRocket stays ahead of LightGBM; CNN vs MiniRocket order changes | `results/figures/robustness_full_N.png`, `src/analysis/weighted.py` |

Planned ablation for M3: the EfficientNet 2 × 2 design (input: spectrogram vs raw signal drawn as an image; weights:
ImageNet vs random), which isolates the effect of pretraining from that of the input format.

### B3. Comparison with papers from A2

| Paper | Their result | Ours | Not equivalent because |
|---|---|---|---|
| [7] Sun et al. 2025 (VIPEEGNet) | KL 0.225 (5-fold CV on HMS) | KL 0.638–0.641 (best models, ≥ 10-vote test windows) | Different evaluation windows and split (their 5-fold CV vs our single frozen fold); 50 s input vs our 10 s; ImageNet pretraining and fine-tuning on ≥ 10-vote windows vs our single-stage training from scratch. The gap cannot be attributed to any one of these |
| [1] Jing et al. 2023 (SPaRCNet) | Above 20–75% of 20 experts on ROC, per class | Per-class AUC at N = 14,215, seed 0: CNN SZ 0.9105, LPD 0.6828, GPD 0.9375, LRDA 0.9234, GRDA 0.8433, other 0.8146 (`results/cnn1d_n14215_s0.json`) | Different metric (they compare with experts' operating points); their test set is 761 other patients, while the HMS patients match their training set; their model was also trained on segments with low-quality labels (p. e1753) |
| [2] Gemein et al. 2020 | Feature-based and deep models within 81–86% accuracy | LightGBM and the CNN are within 0.01 at N = 1,000 and 3,000; at full size the CNN is ahead, with a CI that includes zero | Different task (binary pathology), metric (accuracy) and data; only the qualitative pattern can be compared |

### B4. Timeline

| Task | Owner | Milestone | Planned date | Status (evidence) |
|---|---|---|---|---|
| Proposal, literature matrix, EDA | 20260147S | M1 | Week 3 | Done (`gd1-proposal`, `gd2-dataset-eda`) |
| Preprocessing, leakage audit, baselines, 87-run learning curves | 20260147S | M2 | Week 6 | Done (`gd3-preprocessing`, `gd4-baseline-model`) |
| Progress Test 2 | 20260147S | – | 10 Nov | Done (this file) |
| Fix the deep-learning recipe: patience in gradient steps, learning rate over 3 seeds, recording-weighted KL as a declared second metric | 20260147S | M3 | 15 Nov | Not started (problems documented in `report/M2_preprocessing_and_baselines.md`, Section 6) |
| Proposed models: CBraMod fine-tuning, EfficientNet 2 × 2 (ablation) | 20260147S | M3 (week 10) | 22 Nov | Late: designed (`slides/W7_report_N42.pptx`, slide 11), not run |
| Datasheet and TRIPOD+AI checklist | 20260147S | M3 | 17 Nov | Not started (C1, C2) |
| Logistic regression baseline; calibration curve, calibration-in-the-large, slope, Brier score with 95% CI | 20260147S | M4 (week 12) | 29 Nov | Not started |
| Training methods as separate curves (all windows, augmentation, ≥ 10-vote fine-tuning); explainability and error analysis | 20260147S | M4 (week 12) | 1 Dec | Not started (KL by vote count exists: `results/figures/kl_by_votes_n10000.csv`) |
| IEEE paper, slides | 20260147S | Final (week 15) | 15 Dec | Not started (Lab 1 report as base) |

### B5. Risks, blockers and questions

Risks and blockers:
1. Behind on M3: the proposed models have not been run; one person does every role.
2. The test set is small in patients: 109 patients for the ≥ 10-vote windows, 5 of them hold 35% of the windows, so
   several comparisons cannot be separated. Rotating the test fold would help but multiplies GPU time.
3. Training and scored windows differ (seizure 15% vs 5%): models trained on all windows are judged on the reliable ones.
4. The CNN recipe stops after about 45 gradient steps at N = 100 (patience counted in epochs, `src/models/deep.py:103`),
   and its learning rate was chosen on one noisy run: its small-N points are not valid yet.
5. GPU time on Kaggle is limited (weekly quota).

Questions for the instructor:
1. Is a learning-curve comparison acceptable as the project's main contribution, with the pretrained models (M3) as
   the "proposed models"?
2. Should each milestone be a Pull Request (rubric M2) or only branches on the fork (AI Agents guide)?
3. With six soft-label classes, is one-vs-rest calibration per class acceptable for the M4 calibration items?
4. Could the class project list show the new title (it still shows "ECG DNN adapted to EEG study")?

---

## C. Self-assessment against the appendix (FAIR, TRIPOD+AI, PROBAST+AI)

### C1. FAIR data

| Item | Value |
|---|---|
| Source | HMS Harmful Brain Activity Classification, Kaggle competition (2024): https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification. Built from the MGH data of [1] and [6] |
| Version | No version number on Kaggle. Files used: `train.csv` (dated 10 January 2024 inside the competition archive) and `train_eegs/*.parquet` |
| Download date | Local copy of `train.csv`: 22 September 2026. The Kaggle runs read the competition files directly (run of 23 October 2026) |
| Checksum | SHA-256 of `train.csv`: `e13cc8c7a040c6db5adfb850ee0c2e605365f7e1b7e34ee231c52717886a7886` (computed 10 November 2026). Not yet done for the EEG parquet files, which are read on Kaggle |
| Licence / DUA | Kaggle competition rules, accepted on the Kaggle account; they do not allow redistribution, so no data are in this repository |
| Datasheet | Not yet. Partial content: `README.md` (Data), Lab 1 report Section III. Planned: `docs/datasheet.md` (source, period, population, selection of segments, variables and units, limitations) |
| Patient identifiers | None: `patient_id` is a pseudonymous integer, no names, dates or demographics in the files; the source study states its data were de-identified ([1], p. e1752) |
| F, A, I, R | F: findable (Kaggle page, DOI of [1]); A: access controlled by Kaggle account and rules; I: open formats (CSV, Parquet), 19-electrode 10–20 system at 200 Hz; R: reuse limited by the competition rules, and missing metadata (acquisition dates, devices, demographics) |

### C2. Reporting guideline

| Item | Value |
|---|---|
| Research question | How does the test KL divergence of each model family change with the number of labelled training recordings, when classifying 10 s EEG windows into the expert vote distribution over six patterns, on a patient-disjoint split? |
| Guideline | TRIPOD+AI: the study develops and internally validates diagnostic classification models. Not STARD-AI (no fixed index test evaluated against a reference) and not DECIDE-AI (no clinical deployment) |
| Checklist progress | Not attached yet. Already reported: data source, outcome, predictors, sample sizes, split, missing data (electrodes), model building, performance with CI. Missing: calibration, sample-size justification, fairness by subgroup (no demographics in HMS) |
| Outcome | The distribution of expert votes over seizure, LPD, GPD, LRDA, GRDA and other, for the central 10 s of a 50 s EEG sample |
| Label | Vote proportions (soft label); the majority class is used only for AUC and accuracy |
| Prediction time T0 | The end of the 50 s sample: the signal is filtered on the full 50 s and the models read its central 10 s, so 20 s after the labelled window are used. The models are therefore not causal, as for [7] |
| Variables available at T0 | The 19-electrode EEG of that 50 s sample (16 bipolar chains after preprocessing). No clinical variables, no demographics |

### C3. PROBAST+AI self-appraisal

| Domain | Risk of bias | Evidence in the repository | Mitigation |
|---|---|---|---|
| 1. Participants and data | High | Single centre; windows chosen by the original annotation process, not consecutive EEG (as in [1]); the HMS patients are the training patients of SPaRCNet [1]; the class mix of the split is only approximate: one test patient with 136 recordings makes GRDA 21% of test recordings vs 10% in training (`src/analysis/split.py`) | State the limit; no claim about continuous monitoring; external data (e.g. TUH) only if time allows |
| 2. Predictors | Low | Same preprocessing for every window and model, fixed before the v2 runs (`src/hms_data.py:47`, `:88`); normalisation statistics from training windows only (`src/hms_data.py:292`) | – |
| 3. Outcome | Low to unclear | Outcome = expert votes, defined the same way for all; primary metric on windows with ≥ 10 votes (`src/hms_data.py:48`); many windows have 1 to 5 votes (noisy labels), reported separately (`results/figures/kl_by_votes_n10000.csv`); the reference is the experts' reading of the same EEG, as in [1] | Keep the ≥ 10-vote metric primary; report by vote count |
| 4. Analysis | High | Patient-disjoint split with `assert` (`src/hms_data.py:150`, `:166`) and model choices on validation only (`src/models/lgbm.py:40`, `src/models/minirocket.py:183`, `src/models/deep.py:20`): good. But: no sample-size calculation and only 109 test patients for the main metric; no calibration measures; no logistic regression baseline; no external validation; many model comparisons without correction; the protocol was revised after seeing the first (v1) test results, although v2 uses a new split; the CNN's early stopping is flawed at small N (`src/models/deep.py:103`) | Calibration and logistic regression baseline (M4); fixed CNN recipe before new runs; report CIs and paired tests, not only means; declare the second metric before the next runs |
| Overall | High | Driven by domains 1 and 4 | |

Applicability: the question is about model behaviour as labelled data grow, on selected expert-labelled windows from
one hospital. Results do not transfer to continuous ICU monitoring or to other centres without further validation.

### C4. Quick checklist (appendix Section 7)

The appendix lists nine items; M1 is split into two here to make ten.

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | D1 Datasheet | Not yet | Planned `docs/datasheet.md` (C1) |
| 2 | D2 Version and checksum of the training data | Partly met | SHA-256 of `train.csv` in C1; EEG parquet files not checksummed |
| 3 | D3 No identifiers, legal basis | Met | C1; no data committed (`README.md`, Data) |
| 4 | R1 Guideline chosen, checklist attached | Partly met | TRIPOD+AI chosen (C2); checklist not attached |
| 5 | R2 Outcome, label, T0, variables at T0 | Met | C2 |
| 6 | M1a Split by patient | Met | `src/hms_data.py:150`, `assert` at `:166` |
| 7 | M1b Preprocessing fitted on training data only | Met | `src/hms_data.py:292`; no imputer, no resampling |
| 8 | M2 Logistic regression baseline | Not yet | Baselines are the mean-vote predictor and LightGBM; logistic regression on band power planned (M4) |
| 9 | M3 AUC, calibration curve, calibration-in-the-large, slope, Brier, with 95% CI | Not yet | Only per-class AUC and KL with CI (`src/hms_eval.py:44`, `:28`) |
| 10 | M4 Sample size and applicability | Partly met | No formal sample-size calculation; 109 test patients and their concentration stated (B5); applicability in C3 |

---

## D. AI use declaration

| Tool | Used for | How it was checked |
|---|---|---|
| Claude (Anthropic), through Claude Code | Drafting code (data preparation, models, analysis scripts), analysing the Kaggle results, drafting and editing the reports and this file, checking DOIs on Crossref | I reviewed and ran the code; every number in this file comes from files in `results/` or from Table I, whose entries I checked against the PDFs; DOIs checked on Crossref; study design decisions are mine. Full log: `docs/ai_disclosure_log.md` |
