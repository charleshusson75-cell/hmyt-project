# Milestone 2: preprocessing, leakage control and baselines

Team N42. Data: HMS Harmful Brain Activity Classification [@hms2024], built on the expert-labelled data of Jing et
al. [@jing2023development; @jing2023interrater]. Clinical problem and literature: Lab 1 report, same folder.

## 1. Why the protocol changed after Lab 1

Lab 1 used my first protocol. I then checked every choice against the papers and removed what came from my ECG work
instead of from EEG. Numbers from before and after cannot be compared: windows, split and metric all changed.

| | Before (v1) | Now (v2) | Why |
|---|---|---|---|
| Windows | One random 10 s window per recording | Same for training; validation and test also keep every non-overlapping window | More test windows with ≥ 10 votes; N still counts recordings |
| Signal | 100 Hz, a value set for my ECG inputs | 200 Hz, 0.3–75 Hz band-pass, 60 Hz notch | The recording rate; one signal for every model |
| Split | By patient | By patient and balanced by class | As Jing et al. and Sun et al. do [@jing2023development; @sun2025automated] |
| Main metric | KL on all test windows | KL on test windows with ≥ 10 votes, 95% interval by resampling patients | The high-quality set of Jing et al.; patients, not windows, are independent |
| 1D CNN | TensorFlow, ECG settings, not repeatable on GPU | PyTorch, one recipe for all deep models, repeatable | No setting tuned for ECG |

Checked in the labels (`src/analysis/label_facts.py`): 89.4% of recordings have the same votes in all
their windows, and the 106,800 labelled windows hold only 20,183 distinct (recording, votes) labels. So N counts
recordings, not windows: more windows of a recording add signal, not labels.

## 2. Preprocessing

| Step | Setting | Why | Code |
|---|---|---|---|
| Montage | 16 bipolar chains (longitudinal double banana) from 19 electrodes | The usual montage for reading clinical EEG; removes the common reference | `src/hms_data.py:76` |
| Missing electrode | Its two chains are set to zero; the rest of the window is kept | A missing electrode only affects the chains that use it | `src/hms_data.py:84` |
| Amplitude clip | ± 1,024 µV | Limits extreme artefacts | `src/hms_data.py:45` |
| Notch | 60 Hz | Mains frequency: the data come from a US hospital | `src/hms_data.py:46` |
| Band-pass | 0.3–75 Hz, Butterworth order 3 | Keeps the slow rhythmic delta patterns (LRDA, GRDA) and removes drift; the filter used to pretrain CBraMod, a candidate for the next milestone [@wang2025cbramod] | `src/hms_data.py:47` |
| Zero phase | Forward-backward filtering, on the full 50 s, then the central 10 s | No time shift, and no filter edge effects on the labelled part | `src/hms_data.py:88` |
| Sampling rate | 200 Hz, unchanged | The recording rate; nothing removed before the models | `src/hms_data.py:42` |
| Band power | 160 features: log absolute and relative power, 5 bands × 16 chains, Welch 2 s | Handcrafted features are a strong classical input for EEG [@gemein2020machine] | `src/hms_data.py:104` |
| Normalisation | Per channel, median and IQR, clipped at ± 20 | Robust to artefacts; fitted on training windows only | `src/hms_data.py:292` |

## 3. Zero-leakage audit

| Rule | How it is enforced | Code |
|---|---|---|
| Train and test independent by patient | `StratifiedGroupKFold` on `patient_id`: fold 0 test, fold 1 validation, folds 2–9 training pool | `src/hms_data.py:145` |
| Proof by `assert` | `assert` that train, validation and test share no patient and no recording; runs every time the split is built | `src/hms_data.py:166` |
| Split frozen | Decided once in `prepare()` and saved; every run loads it | `src/hms_data.py:182` |
| Scaling fitted on train only | Median and IQR come from the training windows of that run | `src/hms_data.py:291` |
| No resampling, no global imputation | No SMOTE; a missing chain is filled inside its own window, never from other windows | `src/hms_data.py:84` |
| Windows of one recording stay together | All windows follow their patient into the same split | `src/hms_data.py:156` |

Known limit: patients cannot be split, so class balance is only approximate. One test patient has 136 recordings,
mostly GRDA, which makes GRDA 21% of test recordings against 10% in training (`src/analysis/split.py`, `gd4-baseline-model`).

Baselines and results: branch `gd4-baseline-model`.
