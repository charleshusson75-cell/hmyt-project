"""The 1D CNN of cnn1d.py, then fine-tuned on the training windows with >= 10 votes (two-stage training,
as Sun et al. 2025 do). A training method, not a new model: its curve is read against cnn1d's.

Stage 1 is exactly cnn1d's training (same seed, same initial weights, same recipe), so for a given seed
`val_kl_stage1` equals cnn1d's validation KL. Stage 2 (models/deep.py, finetune_hq) starts from its best
weights and trains only on the windows of the same N training recordings that have >= 10 votes, learning
rate x 0.1; its weights are kept only if the validation KL improves. Same recordings and same windows as
cnn1d: only the weight given to the reliable labels changes. The learning rate is cnn1d's (not tuned again).
"""
NAME = 'cnn1d_ft'
INPUT = 'signal'


def fit_predict(split, seed=0, lr=1e-3, **kw):
    from models import cnn1d
    val_p, test_p, info = cnn1d.fit_predict(split, seed=seed, lr=lr, finetune_hq=True, **kw)
    info['trick'] = 'fine-tuning on the training windows with >= 10 votes'
    return val_p, test_p, info
