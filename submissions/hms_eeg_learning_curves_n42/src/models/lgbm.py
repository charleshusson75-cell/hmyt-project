"""Classical model: LightGBM on the 160 band-power features (F).

It learns from the vote distribution, not from the majority label. Each window is repeated once
per class that received votes, with that class's vote share as the row's weight. The weighted
log-loss of these rows is then exactly the cross-entropy against the vote shares: the same
objective as the CNN.

Trees are added until the validation log-loss (weighted the same way, on the high-quality validation
windows, like the primary metric) stops improving for 100 rounds.
"""
import warnings
import numpy as np

NAME = 'lgbm'
INPUT = 'features'

PARAMS = dict(learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_estimators=2000,
              verbose=-1)


def expand_votes(F, Y):
    """Each window -> one row per class with a non-zero vote share, weighted by that share."""
    Y = np.asarray(Y, np.float64)
    rows, cls = np.nonzero(Y > 0)                   # row-major: window by window, classes in order
    return F[rows], cls, Y[rows, cls]


def fit_predict(split, seed=0, **params):
    import lightgbm as lgb
    p = dict(PARAMS, random_state=seed)
    p.update(params)
    Xe, ye, we = expand_votes(split.F_train, split.Y_train)
    sel = getattr(split, 'val_hq', np.ones(len(split.Y_val), bool))
    Xv, yv, wv = expand_votes(split.F_val[sel], split.Y_val[sel])
    clf = lgb.LGBMClassifier(**p)
    with warnings.catch_warnings():     # LightGBM >= 4.7 renames eval_set; the old name works everywhere
        warnings.filterwarnings('ignore', message=".*'eval_set' is deprecated.*")
        clf.fit(Xe, ye, sample_weight=we, eval_set=[(Xv, yv)], eval_sample_weight=[wv],
                callbacks=[lgb.early_stopping(100, verbose=False)])

    def proba(F):                       # columns of classes absent from training stay at 0
        out = np.zeros((len(F), split.Y_train.shape[1]))
        out[:, clf.classes_.astype(int)] = clf.predict_proba(F)
        return out

    return proba(split.F_val), proba(split.F_test), dict(
        best_iteration=int(clf.best_iteration_ or p['n_estimators']), params=p,
        lightgbm=lgb.__version__)
