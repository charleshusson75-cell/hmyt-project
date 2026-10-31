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

Every choice is made on validation patients, never on test: number of trees (LightGBM), regularisation (MiniRocket,
5 values), learning rate (CNN, 3 values, once at N = 1,000) and best epoch (CNN, every run). TabPFN keeps its defaults.
The test windows are the same in all 87 runs (`test_ids_hash` in each `results/*.json`).

Known limit: patients cannot be split, so class balance is only approximate. One test patient has 136 recordings,
mostly GRDA, which makes GRDA 21% of test recordings against 10% in training (`src/analysis/split.py`).

## 4. Baselines and first comparison

The two baselines are the mean-vote predictor and LightGBM on the 160 band-power features [@ke2017lightgbm]. Three
more families ran on the same split, because the question is how families compare as data grow: TabPFN on the same
features [@hollmann2025accurate], MiniRocket [@dempster2021minirocket] and a 1D CNN (xresnet1d-18 [@strodthoff2021deep])
on the signal. All of them learn from the vote distribution.

Test KL divergence on the windows with ≥ 10 votes (1,484 windows, 109 patients), lower is better, mean (SD) over 3 seeds:

| N | Mean-vote baseline | LightGBM | TabPFN | MiniRocket | 1D CNN |
|---|---|---|---|---|---|
| 100 | 0.936 (0.007) | 0.829 (0.012) | 0.849 (0.054) | 0.857 (0.032) | 1.010 (0.019) |
| 300 | 0.990 (0.013) | 0.905 (0.044) | 0.865 (0.017) | 0.898 (0.016) | 0.963 (0.107) |
| 1,000 | 0.975 (0.005) | 0.859 (0.007) | 0.815 (0.046) | 0.816 (0.029) | 0.867 (0.040) |
| 3,000 | 0.975 (0.005) | 0.831 (0.023) | 0.787 (0.035) | 0.756 (0.034) | 0.824 (0.029) |
| 10,000 | 0.976 (0.002) | 0.809 (0.024) | 0.739 (0.037) | 0.654 (0.008) | 0.702 (0.025) |
| 14,215 | 0.975 (0.000) | 0.813 (0.002) | - | 0.641 (0.005) | 0.638 (0.030) |

With all 14,215 training recordings:

| Model | KL, ≥ 10 votes | 95% interval (seed 0) | KL, one window per recording | Macro AUC (≥ 10 votes) |
|---|---|---|---|---|
| Mean-vote baseline | 0.975 | 0.819 – 1.156 | 1.314 | 0.500 |
| LightGBM | 0.813 | 0.631 – 1.054 | 1.117 | 0.747 |
| MiniRocket | 0.641 | 0.521 – 0.792 | 1.050 | 0.849 |
| 1D CNN | 0.638 | 0.540 – 0.744 | 1.131 | 0.842 |

Figure: `results/figures/learning_curves.png`. All tables: `results/figures/`.

## 5. What this says, and how solid it is

Paired differences below use the same test windows, with a 95% interval by resampling test patients and the 3 seeds
pooled (`results/figures/paired_pooled.csv`).

- N = 100: LightGBM, MiniRocket and TabPFN all beat the CNN, and cannot be separated from each other.
- N = 300 to 1,000: no pair of models is significantly different.
- From N = 3,000: MiniRocket is significantly better than LightGBM. LightGBM stops improving after about 3,000 recordings.
- Full size: CNN and MiniRocket tie. Both are ahead of LightGBM, but only MiniRocket's lead is significant.
- TabPFN is ahead of LightGBM on the same features from 300 to 10,000 recordings, by 0.04 to 0.07, with every interval
  including zero on these windows.
- These compare models, not inputs. Whether the raw signal holds more than band power needs the same model on both.

The main limit is the test set. Its 1,484 windows come from 109 patients, 5 of whom hold 35%. One patient (155 windows,
mostly LRDA) accounts for all of the CNN's lead over LightGBM at full size. Weighting each recording or each patient
equally, or removing that patient (`results/figures/robustness_full_N.png`): MiniRocket stays ahead of LightGBM from
3,000 recordings, but which of CNN and MiniRocket comes first changes. On one window per test recording (any number of
votes), the CNN is not better than LightGBM, even at full size.

## 6. Problems found, and what comes next

1. Training and scored windows differ: seizure is 15% of the votes on training windows, 5% on the scored ones
   (`results/figures/class_mix_train_vs_test.png`). A baseline fitted on ≥ 10-vote windows already gains 0.07.
2. The CNN stops too early with few data: patience counts epochs, so at N = 100 it stops after about 45 gradient
   steps, before reaching the baseline. Its points at N = 100 and 300 measure the stopping rule, not the CNN.
3. The learning-rate choice is weak: it rests on one run whose validation KL jumps by about ± 0.08 between epochs, and
   the chosen value (3 × 10⁻⁴) is the smallest tried.

Planned before the next models: patience counted in gradient steps with a minimum length, learning rate chosen over
3 seeds with 1 × 10⁻⁴ added, and a second metric declared in advance (KL with each test recording weighted equally).
Then pretrained models (CBraMod [@wang2025cbramod], an ImageNet CNN [@tan2021efficientnetv2] as in Sun et al.
[@sun2025automated]) and training methods as separate curves.

## AI and AI-Assisted Technologies Disclosure

I used Claude (Anthropic), through Claude Code, to draft code, analyse results and draft and edit text; each use is
listed in `docs/ai_disclosure_log.md`. I reviewed the code and ran it, the study design decisions are mine, and I take
full responsibility for this report.

References: `report/references.bib`.
