# Biomedical Reproducibility Checklist (team N42, final submission)

The checklist from the course's Pull Request template, filled in for `gd7-ieee-paper`.

- [x] Public, licensed data; source and link in `README.md`; no identifying patient data in this repository
- [x] Split by patient (10 class-stratified patient folds), with `assert` proving train, validation and test share no patient (`src/hms_data.py`, `make_split` and `check_disjoint`)
- [x] Scaling fitted on training windows only (median/IQR per chain); no SMOTE, no imputer, no feature selection
- [x] Random seeds fixed; deterministic CNN kernels (69 runs repeated three weeks apart gave identical predictions)
- [x] `requirements.txt` pins the versions the Kaggle runs recorded (the others were not recorded); commands in `README.md`
- [x] Sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier score, each with a 95% interval over patients; calibration curves, calibration-in-the-large and slope (`results/final_cv/analysis/m4_common*.csv`, paper Table III, Fig. 5)
- [x] XAI (LightGBM SHAP, CNN occlusion and Grad-CAM on 3 correct and 3 misclassified cases shown, 6 + 6 in `results/final_cv/figures`) and error analysis (confusion matrix, errors by class, votes, disagreement, patients)
- [x] Every number in the paper traces to `results/final_cv/` (`analysis/report_numbers.py` and the notebook's tables)
- [x] Citations in `report/references.bib`, DOIs checked on Crossref
- [x] AI disclosure section in the paper, and `docs/ai_disclosure_log.md`
