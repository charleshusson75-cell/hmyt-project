# Cross-validation over the 10 patient folds (Kaggle, 17–19 November 2026)

Three Kaggle sessions of `hms_curve.ipynb`, each attaching the output of the one before:

| session | what ran | output downloaded as |
|---|---|---|
| 1 (17–18 Nov) | fold 0 reused from the recipe-2 run (`results/m3_recipe2`); CV folds 1–5 | `hms_results-4` |
| 2 (18 Nov) | CV folds 6–9; pooled tables (section 9), Milestone 4 (section 10), explainability (section 11) | `hms_results-5` |
| top-up (19 Nov) | retry of the 4 failed runs (failed again); explainability with misclassified cases (`xai_cnn_gradcam_errors.png`, `xai_cnn_cases.csv`) | `hms_results-6` |

This folder = session 2's output plus the two new files of the top-up (every other file of the top-up was
byte-identical to session 2, including the retrained CNN's Grad-CAM: the float32 fallback of D-56 changed nothing).

## Content

- Fold 0 (main split, 3 seeds): `*.json`, `preds/` at the root, 140 of 141 runs.
- Folds 1–9 (seed 0): `cv/fold_k/`, 47 runs each except fold 1 (45) and fold 2 (46).
- Notebook tables and figures: `figures/` (`cv_*` pooled, `m4_*` Milestone 4, `xai_*` explainability).
- `analysis/report_numbers.py` → every number of the final paper and slides (CSV tables next to it);
  `analysis/report_figures.py` → the report figures in `analysis/figures/`. Run with the results folder and the
  code folder as arguments.

**560 of 564 planned runs.** Missing, failed in every session: fold 0 `cnn1d_ft` N = 1,000 seed 2; fold 1 `cnn1d`
and `cnn1d_ft` with all recordings; fold 2 `cnn1d_ft` N = 3,000. The float16 hypothesis (D-56) did not fix them;
the cause is unknown (the error text was only in the Kaggle log, and the Kaggle copy of the notebook did not have
the new `failures.txt` logging). Affected cells are pooled over 9 folds; paired tests use the folds both models have.

## Findings (test KL on windows with ≥ 10 votes, 16,255 windows, 1,053 patients)

| N | best | paired difference with the runner-up [95% CI] | folds won |
|---|---|---|---|
| 100 | LR, band power (0.906) | −0.040 [−0.082, −0.004] vs LightGBM | 7/10 |
| 300 | LR, MiniRocket + band (0.863) | −0.005 [−0.035, 0.029] vs LR band | 3/10 |
| 1,000 | LR, MiniRocket + band (0.780) | −0.016 [−0.055, 0.028] vs TabPFN | 7/10 |
| 3,000 | LR, MiniRocket + band (0.732) | −0.007 [−0.060, 0.049] vs CNN + ft | 4/10 |
| 10,000 | CNN + fine-tuning (0.566) | −0.119 [−0.161, −0.080] vs LR, MiniRocket + band | 10/10 |
| all | CNN + fine-tuning (0.576, 9 folds) | −0.087 [−0.118, −0.063] vs CNN | 8/9 |

The KL of the best model is from `kl_table.csv` (every fold where it ran); the paired difference uses only the folds
where both models ran. They differ only where a run failed: at 3,000 the LR on MiniRocket + band power scores 0.732 over
10 folds and 0.744 over the 9 folds shared with the CNN + fine-tuning (`best_by_N.csv`, `kl_table_common.csv`).
TabPFN has no row with all recordings: it was not run there (pretrained for at most 10,000 training samples), so it is
also absent from the Milestone 4 tables (`m4_common*.csv`).

- CNN from scratch: worse than the mean-vote baseline at N = 100 (1.265 vs 1.054), beats it from 1,000.
- Same learner (LR): band power better than MiniRocket features at 100 (−0.054), worse from 1,000 (+0.05 to +0.13);
  both inputs together slightly better than MiniRocket alone (significant at 5 of 6 sizes).
- Fine-tuning on ≥ 10-vote windows: kept in 1 of 20 runs at N ≤ 300 (no effect), 28 of 38 from 1,000 (−0.06 to −0.12).
  Without it the CNN only ties the LR on MiniRocket + band power at ≥ 10,000 — its lead depends on a training method
  the feature models did not get.
- TabPFN = LightGBM at 100–300, better from 1,000 (−0.04). Fold 0 alone had LightGBM just better at 100
  (0.020 [0.0002, 0.042]); over 10 folds 0.014 [−0.006, 0.034]: did not replicate.
- Fold-to-fold spread (CNN + ft, all recordings: 0.45–0.66) ≫ seed spread on fold 0 (SD 0.03).
- Milestone 4 (all recordings, 9 common folds, 14,793 windows): CNN + ft best on KL 0.576, Brier 0.213, macro AUC
  0.868, PR-AUC 0.560, F1 0.488, but seizure sensitivity 0.29 at the most likely class (LightGBM 0.59, which predicts
  a seizure share of 0.17 where experts gave 0.05; seizure AUC 0.835 vs 0.816). LPD and LRDA mostly called "other";
  windows with 3–5 votes hardest (KL 1.34); top 10 patients = 23% of the error for 15% of the windows.
- Explainability (fold 0): LightGBM SHAP — GPD theta 72%, GRDA delta 40%, seizure alpha + beta 77%; CNN occlusion —
  F3–C3 matters most for GPD and GRDA; Grad-CAM marks each GPD discharge; errors: a focal right-posterior seizure
  called "other", a false seizure on a transient, an O1/O2-artifact window labelled half GRDA by the experts.

Final report: `report/N42_final_paper.pdf` and `slides/final_report_N42.pptx`, built from these files.
