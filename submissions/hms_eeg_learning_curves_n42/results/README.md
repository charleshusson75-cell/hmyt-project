# Learning curves, protocol v2 (Kaggle run of 23 October 2026, step A)

All 87 runs come from one Kaggle GPU session of `hms_curve.ipynb` (protocol v2): baseline, LightGBM, CNN
and MiniRocket at N = 100, 300, 1,000, 3,000, 10,000 and 14,215 (all training recordings), 3 seeds; TabPFN
up to 10,000. No failed run. Every run has the same test set (hash `b96aae4165ee`). The split was rebuilt
locally from `train.csv`: it gives exactly the same 2,512 test windows.

Numbers here cannot be compared with the v1 folders (windows, split, sampling rate and metric all changed).

- `tuning.json`: the CNN learning rate, chosen on validation at N = 1,000, seed 0: **3e-4** (val KL 0.732),
  1e-3 (0.766), 3e-3 (0.791). The tuning run and the curve run `cnn1d_n1000_s0` ran in different processes
  and give the same validation KL to 5 decimals: the CNN is now repeatable (v1's was not).
- `figures/`: what the notebook saved. `learning_curves*.png` were redrawn locally with the same data after
  a label fix in `plot_curve` (two end labels overlapped). `robustness_full_N.png` and
  `class_mix_train_vs_test.png` were drawn locally (`../src/analysis/figs.py`).
- `../src/analysis/`: the scripts behind the numbers below (`analyse.py`: split check, class mix, pooled paired
  tests, per-patient breakdown; `weighted.py`: other weightings; `split.py`: split balance).

## Test sets

- **hq** (main metric): every stored test window with ≥ 10 votes, 1,484 windows from 557 recordings and
  **only 109 patients**; the 5 largest patients hold 35% of these windows, one patient alone 155 (10%).
- **all**: one window per test recording (1,564), any number of votes.

## Test KL on the hq windows (mean over 3 seeds, SD in brackets; lower is better)

| N | baseline | LightGBM | TabPFN | MiniRocket | CNN |
|---|---|---|---|---|---|
| 100 | 0.936 | **0.829** (0.012) | 0.849 (0.054) | 0.857 (0.032) | 1.010 (0.019) |
| 300 | 0.990 | 0.905 (0.044) | **0.865** (0.017) | 0.898 (0.016) | 0.963 (0.107) |
| 1,000 | 0.975 | 0.859 (0.007) | **0.815** (0.046) | 0.816 (0.029) | 0.867 (0.040) |
| 3,000 | 0.975 | 0.831 (0.023) | 0.787 (0.035) | **0.756** (0.034) | 0.824 (0.029) |
| 10,000 | 0.976 | 0.809 (0.024) | 0.739 (0.037) | **0.654** (0.008) | 0.702 (0.025) |
| 14,215 | 0.975 | 0.813 (0.002) | — (limit 10,000) | 0.641 (0.005) | **0.638** (0.030) |

On the **all** windows the order is different: TabPFN is best at 10,000 (1.024), MiniRocket at 14,215
(1.050), and the CNN is no better than LightGBM even at full size (1.131 vs 1.117). Table:
`figures/test_kl_all_table.csv`.

## Paired tests (3 seeds pooled, 95% interval by resampling test patients; `figures/paired_pooled.csv`)

- **N = 100:** LightGBM, MiniRocket and TabPFN all beat the CNN; they cannot be separated from each other.
- **N = 300–1,000:** no pair of models is significantly different.
- **N = 3,000:** MiniRocket better than LightGBM (−0.075 [−0.122, −0.018]); not separable from CNN or TabPFN.
- **N = 10,000:** MiniRocket better than LightGBM (−0.155) and TabPFN (−0.085 [−0.126, −0.037]); not
  separable from the CNN.
- **N = 14,215:** CNN and MiniRocket tie (−0.003 [−0.145, 0.124]). MiniRocket beats LightGBM (−0.172
  [−0.283, −0.061]); the CNN's gap to LightGBM (−0.175) has an interval that includes zero [−0.414, 0.040].
- **TabPFN vs LightGBM (same features):** TabPFN ahead from 300 to 10,000 by 0.04–0.07, intervals include
  zero on hq; significant at 10,000 on the all windows (−0.089 [−0.128, −0.049]).
- `figures/paired_table.csv` (the notebook's version) counts significant seeds one by one; same picture.

As before: these compare models, not inputs.

## How solid (full N, same predictions averaged differently; `figures/robustness_full_N.png`)

| weighting | LightGBM | MiniRocket | CNN |
|---|---|---|---|
| per window (main) | 0.813 | 0.641 | 0.638 |
| per recording | 0.685 | 0.577 | 0.584 |
| per patient | 0.740 | 0.653 | 0.603 |
| without patient 56885 (155 hq windows, 149 LRDA) | 0.680 | 0.562 | 0.628 |

- Holds under every weighting: MiniRocket ahead of LightGBM from N = 3,000; the CNN worst at N ≤ 300;
  LightGBM flat after about 3,000 while MiniRocket and the CNN keep improving.
- Does not hold: which of CNN and MiniRocket is first at full N. Patient 56885 accounts for more than all of
  the CNN's lead over LightGBM at full N (CNN better on 78 of 109 patients, but the size of the gap comes
  from this one patient).

## Problems the run revealed

1. **Training and scored windows differ.** Mean vote share of seizure: 15% on the training windows (what
   the models see), 5% on the hq test windows. A baseline fitted on the training hq windows scores 0.908 on
   test hq instead of 0.975. Fits Sun et al.'s two-stage training (fine-tune on ≥ 10-vote windows), step C.
2. **The CNN stops too early at small N.** Patience is 15 epochs: at N = 100 (2 steps per epoch) training
   ends after 20–24 epochs, about 45 gradient steps; its best validation KL is 1.04–1.08, worse than the
   baseline's 0.83–0.88. The N = 100–300 CNN points measure the stopping rule, not the CNN.
3. **The learning-rate choice is weak.** It rests on one run (seed 0) whose validation KL jumps by about
   ±0.08 between epochs; 3e-4 won on a single good epoch (0.732, neighbours 0.82–0.92) and is the smallest
   value of the grid.
4. **Noisy best-epoch selection.** At full N the validation KL moves by ±0.08 from one epoch to the next
   while the training loss falls steadily; the best epoch (10–14) is partly luck.
5. **Split balance.** StratifiedGroupKFold cannot split patients: one test patient has 136 recordings (97
   GRDA), so GRDA is 21% of test recordings vs 10% in training; LPD 7% vs 17%.
6. MiniRocket's C sits at the lowest value of the grid (1e-4) in 5 of 9 runs at N ≤ 1,000.

Proposed before step B (not done): patience counted in gradient steps with a minimum length; learning rate
over 3 seeds with 1e-4 added; a recording-weighted KL declared as a second metric before new runs.

## Run times (seconds per run, Kaggle GPU)

LightGBM 2–10; TabPFN 20 (N = 100) to 466 (N = 10,000); MiniRocket 56–210; CNN 13–242 (best epoch
10–27). Plus loading the data in each run's own process.
