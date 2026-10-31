# AGENTS.md (team N42)

- Topic: How Many Labelled EEG Windows Does Each Model Family Need? A Learning-Curve Study of Harmful Brain Activity Classification
- Data: HMS Harmful Brain Activity Classification, public on Kaggle: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification

Rules for any coding agent working in this folder:
- Work only inside this folder.
- Never read, upload or print real patient data; no API keys or `.env` files.
- Split by patient before any scaling or imputation, and fit preprocessing on train only.
- Never invent references, numbers or results. Mark unverified items `[TODO: verify]`.
- Fixed random seeds and logged runs.
- Before finishing a task: run the code, show the diff, list what was not tested.
