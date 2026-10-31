# Biomedical Reproducibility Checklist (team N42, Milestone 2)

The checklist from the course's Pull Request template, filled in for `gd4-baseline-model`. Unticked items belong to later milestones.

- [x] Public, licensed data; source and link in `README.md`; no identifying patient data in this repository
- [x] Split by patient, with `assert` proving train and test share no patient (`src/hms_data.py`, `check_disjoint`)
- [x] Scaling fitted on training data only; no SMOTE, no imputer, no feature selection
- [x] Random seeds fixed
- [x] `requirements.txt` pins the versions the Kaggle run recorded (the others were not recorded); commands in `README.md`
- [ ] Sensitivity, specificity, F1, ROC-AUC, PR-AUC with 95% CI. So far: KL with 95% interval and AUC per class. Milestone 4.
- [ ] XAI and error analysis. So far: KL by number of votes and per class. Milestone 4.
- [x] Every number in the report traces to `results/` or `src/analysis/`
- [x] Citations in `report/references.bib`, DOIs checked on Crossref
- [x] AI disclosure section in the Milestone 2 report, and `docs/ai_disclosure_log.md`
