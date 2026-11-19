# Learning curves after the recipe fix (Kaggle run of 17 November 2026)

140 runs from one Kaggle GPU session of `hms_curve.ipynb` with the code of 10 November (decisions D-48 to D-51):
CNN recipe 2 (`cnn1d`), CNN + fine-tuning on the >= 10-vote windows (`cnn1d_ft`), logistic regression on band power
(`logreg_band`) and on MiniRocket features + band power (`logreg_both`), and the earlier models recomputed.
Same test set as 23 October (hash `b96aae4165ee`).

- **Missing:** `cnn1d_ft`, N = 1,000, seed 2 (no result written; cause in the notebook log, not in the zip).
- **Reproducibility:** baseline, LightGBM, MiniRocket and TabPFN were recomputed (the 23 October output was not
  attached) and give bit-identical predictions to 23 October, in all 69 runs.
- **Learning rate (recipe 2):** 1e-3, mean validation KL over 3 seeds 0.751; 3e-4: 0.758, 3e-3: 0.763, 1e-4: 0.854
  (`tuning.json`). 1e-3 and 3e-4 are practically tied.

## Test KL on the windows with >= 10 votes (mean over 3 seeds, SD in brackets; lower is better)

| N | baseline | LightGBM | log. reg., band power | MiniRocket | log. reg., both | TabPFN | CNN | CNN + fine-tuning |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.936 | 0.829 (0.012) | **0.810** (0.007) | 0.857 (0.032) | 0.852 (0.032) | 0.849 (0.054) | 1.217 (0.189) | 1.217 (0.189) |
| 300 | 0.990 | 0.905 (0.044) | 0.874 (0.008) | 0.898 (0.016) | 0.885 (0.024) | **0.865** (0.017) | 1.003 (0.105) | 1.003 (0.105) |
| 1,000 | 0.975 | 0.859 (0.007) | 0.825 (0.005) | 0.816 (0.029) | **0.800** (0.033) | 0.815 (0.046) | 0.819 (0.056) | 0.843 (0.054)* |
| 3,000 | 0.975 | 0.831 (0.023) | 0.809 (0.003) | 0.756 (0.034) | 0.739 (0.033) | 0.787 (0.035) | 0.781 (0.028) | **0.723** (0.076) |
| 10,000 | 0.976 | 0.809 (0.024) | 0.819 (0.002) | 0.654 (0.008) | 0.642 (0.010) | 0.739 (0.037) | 0.635 (0.034) | **0.549** (0.026) |
| 14,215 | 0.975 | 0.813 (0.002) | 0.812 (0.000) | 0.641 (0.005) | 0.631 (0.005) | – | 0.614 (0.047) | **0.535** (0.031) |

\* 2 seeds; for these two seeds fine-tuning was not kept (identical to CNN), so the gap is the missing seed.
One window per test recording: `figures/test_kl_all_table.csv`. Paired tests (seeds pooled, bootstrap over test
patients): `figures/paired_pooled.csv`.

## Findings

1. **Same learner, different input** (logistic regression; `figures/learning_curves_input_same_learner.png`).
   Band power is as good as MiniRocket features up to 1,000 recordings (at N = 100 slightly better, interval just
   includes zero). MiniRocket features win from 10,000 (−0.165 and −0.171, significant; at 3,000 the interval just
   includes zero). Band power stops improving after ~1,000 recordings with this learner, as with LightGBM.
   Adding band power to MiniRocket features helps at every N, significantly but slightly (0.005–0.017).
2. **Same input, different learner.** Logistic regression and LightGBM on band power are equivalent (one
   significant difference in 6 sizes, at N = 300, for logistic regression). Together with 1, the gap between
   MiniRocket and LightGBM at large N comes from the input, not the learner.
3. **CNN, recipe 2.** From 1,000 recordings its mean is lower than with recipe 1 at every size (0.819 vs 0.867 …
   0.614 vs 0.638). Two things changed (stopping rule and learning rate), so the gain is not attributed to one.
   At 100 and 300 recordings it is worse (1.217, 1.003) and above the baseline: it memorises the training set
   (training loss ~0.36) while validation KL stays noisy around 0.9–1.3, and the best of ~100 validation checks is
   picked partly by luck (validation looks better than test). With either recipe the CNN trained from scratch does
   not beat the baseline below ~1,000 recordings: this, not the stopping rule alone, explains the small-N points.
4. **Fine-tuning on the >= 10-vote windows.** Kept in 6 of 6 runs from 10,000 and 2 of 3 at 3,000; never at
   1,000 or below (29–324 such windows). Gain on the >= 10-vote test windows: −0.058 (3,000), −0.086 (10,000),
   −0.079 (14,215), all significant. CNN + fine-tuning has the lowest mean from 3,000 recordings (not significant at 3,000), and at full size it
   stays first under all four averages (per window 0.535, per recording 0.468, per patient 0.484, without patient
   56885 0.499; `analysis/check.py`). Against MiniRocket: −0.106 (10,000) and −0.107 (14,215), significant.
   Cost: on the windows with fewer votes it is worse (one window per recording: 1.094 vs 1.036 at full size).
   Consistent with the intent: it lowers the predicted seizure share on the >= 10-vote windows from 0.112 to
   0.079 (true 0.047), and also on the low-vote windows, where seizure is common (0.115 to 0.064; true 0.267).
5. **Unchanged models** (LightGBM, TabPFN, MiniRocket): same numbers as 23 October.

Earlier statements that need correcting in the reports: "the CNN's points at N = 100–300 measure the stopping
rule, not the CNN" (finding 3); at N = 100 LightGBM is just significantly better than TabPFN (interval 0.0002 to
0.042), not "cannot be separated".
