"""Logistic regression on the 160 band-power features: MiniRocket's head on a different input.

Two uses. (1) The logistic-regression baseline that PROBAST+AI asks for (course appendix, item M2).
(2) The same-learner test of the input: one head (multinomial logistic regression on the vote shares,
L2 strength from the same grid, chosen on the high-quality validation windows, models/minirocket.py
select_head) on band power (this file), on MiniRocket features (minirocket.py) and on both
(logreg_both.py). Differences between these three curves come from the input only, unlike the
comparison of LightGBM on band power with MiniRocket, where the learner changes too.

Features standardised with the training windows' statistics only.
"""
import numpy as np

NAME = 'logreg_band'
INPUT = 'features'


def fit_predict(split, seed=0, Cs=None, max_iter=500):
    from models.minirocket import select_head, standardise, CS
    Ftr, Fva, Fte = standardise(*(np.asarray(a, np.float32) for a in (split.F_train, split.F_val, split.F_test)))
    pv, pt, info = select_head(Ftr, Fva, Fte, split, Cs or CS, max_iter)
    info['features'] = 'band power (%d)' % Ftr.shape[1]
    return pv, pt, info
