"""Baseline: predict the mean vote distribution of the training windows, for every window.
Any model that does not beat this has learnt nothing from the signal."""
import numpy as np

NAME = 'floor'
INPUT = 'none'


def fit_predict(split, seed=0):
    q = split.Y_train.mean(0, dtype=np.float64)
    return (np.tile(q, (len(split.Y_val), 1)), np.tile(q, (len(split.Y_test), 1)),
            dict(prior=[round(float(v), 4) for v in q]))
