"""TabPFN v2 (Hollmann et al., Nature 2025) on the 160 band-power features (F).

A transformer pretrained on millions of synthetic tables that predicts "in context": at prediction
time it receives the N training windows with their labels and the windows to predict, in one forward
pass. Nothing is trained on HMS; the training windows are its input. Preprocessing is TabPFN's own and
is fitted on the training windows only.

Votes: TabPFN takes one label per window. Each window's label is drawn from its vote shares, DRAWS
times with different seeds, and the predicted probabilities are averaged, so on average the labels it
sees are the vote shares. TabPFN's default ensemble of 8 members is split over the draws (4 draws x 2
members): the cost of one default TabPFN.

A class can be missing from a draw (TabPFN then gives it probability 0, and the KL of a window that has
votes for it would be infinite), so the averaged probabilities are mixed with the training prior at
weight 1e-4: a negligible change everywhere else.

Limit: pretrained for up to 10,000 samples (Hollmann et al., Methods), so the curve stops at N = 10,000
(MAX_N; run.py skips larger sizes). Needs tabpfn==2.1.3 (the version TSPFN builds on); its weights are
downloaded on first use, so Kaggle needs Internet on.
"""
import time
import numpy as np

NAME = 'tabpfn'
INPUT = 'features'
MAX_N = 10_000
DRAWS, MEMBERS = 4, 2


def draw_labels(Y, rng):
    """One class per row, drawn with probabilities = the row's vote shares."""
    Y = np.asarray(Y, np.float64)
    cum = np.cumsum(Y / Y.sum(1, keepdims=True), 1)
    return np.minimum((rng.random((len(Y), 1)) > cum).sum(1), Y.shape[1] - 1)


def predict(clf, F, chunk=500):
    """predict_proba, in chunks of `chunk` windows if the GPU runs out of memory (N close to 10,000)."""
    try:
        return clf.predict_proba(F)
    except RuntimeError as e:                          # torch.OutOfMemoryError is a RuntimeError
        if 'out of memory' not in str(e).lower():
            raise
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate([clf.predict_proba(F[i:i + chunk]) for i in range(0, len(F), chunk)])


def fit_predict(split, seed=0, draws=DRAWS, members=MEMBERS, eps=1e-4, device='auto'):
    import torch, tabpfn
    from tabpfn import TabPFNClassifier
    rng = np.random.default_rng(seed)
    k = split.Y_train.shape[1]
    pv, pt = np.zeros((len(split.F_val), k)), np.zeros((len(split.F_test), k))
    missing, t0 = [], time.time()
    for d in range(draws):
        y = draw_labels(split.Y_train, rng)
        clf = TabPFNClassifier(n_estimators=members, random_state=seed * 100 + d, device=device)
        clf.fit(split.F_train, y)
        cols = clf.classes_.astype(int)
        pv[:, cols] += predict(clf, split.F_val)
        pt[:, cols] += predict(clf, split.F_test)
        missing.append(int(k - len(cols)))
    prior = split.Y_train.mean(0, dtype=np.float64)
    pv, pt = ((1 - eps) * P / draws + eps * prior for P in (pv, pt))
    return pv, pt, dict(draws=draws, members_per_draw=members, classes_missing_per_draw=missing,
                        prior_mix=eps, device=str(clf.device_) if hasattr(clf, 'device_') else device,
                        tabpfn=getattr(tabpfn, '__version__', '2.1.3'), seconds_tabpfn=round(time.time() - t0, 1))
