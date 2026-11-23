# AI disclosure log (team N42)

Written on 2026-10-31 from my project notes, when the GitHub process was set up; new lines are added as work goes on.

| Date | Member (student ID) | Tool | Task | What we checked / changed |
|---|---|---|---|---|
| 2026-10-07 | 20260147S | Claude (Claude Code) | Project plan for the learning-curve study | Kept to the framing the instructor approved on 21 September; an earlier, too complex code base was set aside |
| 2026-10-08 to 10-11 | 20260147S | Claude (Claude Code) | Data preparation, first CNN and LightGBM scripts, pilot notebook | Read the code, ran it on Kaggle, looked at the pilot results before going on |
| 2026-10-12 to 10-13 | 20260147S | Claude (Claude Code) | Lab 1: search for candidate papers and DOIs, finding passages in the PDFs for the matrix, EDA code, drafting and English editing | Checked every matrix entry against the PDF; reviewed and ran the code; reviewed and corrected the text |
| 2026-10-15 to 10-16 | 20260147S | Claude (Claude Code) | MiniRocket port, TabPFN wrapper, analysis of the curves | MiniRocket transform compared with aeon by tests; rejected the claim "raw signal beats band power", which compared different models |
| 2026-10-17 | 20260147S | Claude (Claude Code) | Week 6 slides | Read them; asked for a shorter deck |
| 2026-10-23 | 20260147S | Claude (Claude Code) | Review of each design choice against the papers; protocol v2 code; curve notebook | Decided the windows, the split and keeping the CNN myself; split checked to be patient-disjoint; repeated runs give identical CNN results; ran it on Kaggle |
| 2026-10-23 | 20260147S | Claude (Claude Code) | Analysis of the Kaggle results; Week 7 slides | Numbers come from the files of the run; read the analysis and slides |
| 2026-10-31 | 20260147S | Claude (Claude Code) | This submission: folder, copies without my name, README, Milestone 2 report draft, `references.bib`, `assert` in the split, Git commits | Course validator and review tool run; DOIs checked on Crossref; I read the README, the report and this log before pushing |
| 2026-11-10 | 20260147S | Claude (Claude Code) | Progress Test 2 file (`report/kiem_tra_lan_2.md`): structure, tables from Table I and `results/`, PROBAST+AI drafts | Numbers checked against `results/` and the PDFs; I read the file and corrected it before pushing |
| 2026-11-10 to 11-19 | 20260147S | Claude (Claude Code) | Recipe 2, same-learner LR models, fine-tuning, 10-fold cross-validation, Milestone 4 metrics and explainability code; Kaggle session instructions; analysis of the results | Chose which improvements to make and the 10-fold design; ran every Kaggle session; numbers checked against the result files; a wrong claim (CNN small-N results as a stopping-rule artefact) and a wrong direction (TabPFN vs LightGBM on fold 0) were caught and corrected |
| 2026-11-19 | 20260147S | Claude (Claude Code) | Final paper (IEEE style) and final slides: drafting, figures, tables | Every number traced to `results/kaggle_2026-11-18_cv/analysis/`; I read and edited the paper and slides before submitting |
| 2026-11-23 | 20260147S | Claude (Claude Code) | Instructor review questions: checks against the files, clarifications in the README, the results README, `src/eda/` and the paper (TabPFN with all recordings, which table each number comes from, Lab 1 vs final preprocessing), rebuilt paper, draft of the reply | Answers checked against the code and the result tables; I read the changes before pushing |
