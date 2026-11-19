"""Logistic regression on MiniRocket features and band power together: the third arm of the
same-learner test (see logreg_band.py).

The MiniRocket part is exactly minirocket.py's (same seed, same parameters, same features); the 160
band-power features are appended. Each feature is standardised with the training windows' statistics,
so every feature, from either input, starts with the same weight in the L2 penalty. Same head, same
grid of C, same selection on the high-quality validation windows as minirocket.py and logreg_band.py.
"""
import numpy as np

NAME = 'logreg_both'
INPUT = 'signal'                    # needs the signal for MiniRocket; band power comes with every split


def fit_predict(split, seed=0, n_features=10_000, Cs=None, max_iter=500):
    from models.minirocket import features, select_head, standardise, CS
    Mtr, Mva, Mte, params, t_transform = features(split, seed, n_features)
    Btr, Bva, Bte = standardise(*(np.asarray(a, np.float32) for a in (split.F_train, split.F_val, split.F_test)))
    n_mr = Mtr.shape[1]
    Ftr, Fva, Fte = (np.concatenate(p, 1) for p in ((Mtr, Btr), (Mva, Bva), (Mte, Bte)))
    del Mtr, Mva, Mte
    pv, pt, info = select_head(Ftr, Fva, Fte, split, Cs or CS, max_iter)
    info.update(features='MiniRocket (%d) + band power (%d)' % (n_mr, Btr.shape[1]),
                dilations=[int(d) for d in params['dilations']], transform_seconds=t_transform)
    return pv, pt, info
