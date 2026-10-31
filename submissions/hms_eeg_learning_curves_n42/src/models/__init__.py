"""One file per model. Each exposes:

    NAME  = 'xresnet1d'
    INPUT = 'signal' | 'features' | 'none'
    fit_predict(split, seed=0, **kw) -> (val_proba, test_proba, info)

Probabilities have shape (n, 6) and rows summing to 1. Optional: setup(), called before
the data are loaded. Adding a model = adding one file here.
"""
