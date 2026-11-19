"""Milestone 4: clinical metrics with 95% intervals, calibration, and error analysis, from saved predictions.

    Y (n, 6) expert vote shares, P (n, 6) predicted probabilities, patients (n,)
    table = metrics_table(Y, P, patients)          every metric with a 95% interval over patients
    plot_calibration(Y, {'CNN': P1, 'LightGBM': P2})

Labels are vote shares, not single classes, so:
- discrimination metrics (AUC, PR-AUC, sensitivity, specificity, F1, accuracy) use the majority class of
  each window as the label, one class against the rest;
- calibration compares, per class, the predicted probability with the observed SHARE of votes for that class:
  a window predicted "30% seizure" is well calibrated if, among such windows, 30% of the votes are seizure;
- calibration-in-the-large is the intercept of a logistic recalibration with the slope fixed at 1 (0 = right on
  average; > 0 = the class is under-predicted), the calibration slope its slope (1 = ideal; < 1 = predictions too
  extreme). Both fitted on the soft labels by maximum likelihood;
- the Brier score is the squared distance between the predicted and the voted distribution, summed over classes.
Intervals: bootstrap of whole patients (all windows of a drawn patient), as everywhere in this project.
Plain numpy: no scikit-learn needed.
"""
import numpy as np
from hms_data import CLASSES
from hms_eval import kl_per_sample

_EPS = 1e-6


# ----------------------------------------------------------------------------- one-vs-rest building blocks
def _rank(a):
    """1-based ranks, ties get their average rank."""
    order = np.argsort(a, kind='mergesort')
    _, first, counts = np.unique(a[order], return_index=True, return_counts=True)
    ranks = np.empty(len(a))
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    return ranks


def auc(pos, score):
    """ROC AUC of `score` for the boolean labels `pos` (Mann-Whitney form)."""
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((_rank(score)[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(pos, score):
    """Area under the precision-recall curve, step-wise, one step per distinct score (tied scores form one
    threshold): the definition of scikit-learn's average_precision_score."""
    n1 = int(pos.sum())
    if n1 == 0:
        return np.nan
    order = np.argsort(-score, kind='mergesort')
    s, h = score[order], pos[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]       # end of each group of tied scores
    tp = np.cumsum(h)[last]
    fp = (last + 1) - tp
    return float(np.sum(np.diff(np.r_[0.0, tp / n1]) * tp / (tp + fp)))


def recalibration(y, p):
    """Logistic recalibration of predicted probabilities p on observed shares y (both in [0, 1]):
    returns (calibration-in-the-large, slope). CITL: intercept a of y ~ sigmoid(a + logit p);
    slope: b of y ~ sigmoid(a' + b logit p). Newton's method on the soft-label log-likelihood."""
    z = np.log(np.clip(p, _EPS, 1 - _EPS) / (1 - np.clip(p, _EPS, 1 - _EPS)))
    sig = lambda t: 1.0 / (1.0 + np.exp(-t))
    a = 0.0
    for _ in range(50):                          # intercept only, slope fixed at 1
        q = sig(a + z)
        g, h = (q - y).sum(), (q * (1 - q)).sum() + 1e-9
        a -= g / h
        if abs(g / h) < 1e-8:
            break
    w = np.zeros(2)
    X = np.stack([np.ones_like(z), z], 1)
    for _ in range(100):                         # intercept and slope
        q = sig(X @ w)
        g = X.T @ (q - y)
        H = (X * (q * (1 - q))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(w[1])


# ----------------------------------------------------------------------------- all metrics of one set
def metrics(Y, P):
    """Every metric of the table for one set of windows: a flat dict."""
    Y, P = np.asarray(Y, np.float64), np.asarray(P, np.float64)
    true, pred = Y.argmax(1), P.argmax(1)
    out = dict(kl=float(kl_per_sample(Y, P).mean()), brier=float(((P - Y) ** 2).sum(1).mean()),
               accuracy=float((true == pred).mean()))
    for j, c in enumerate(CLASSES):
        pos, hit = true == j, pred == j
        tp, fp, fn = int((pos & hit).sum()), int((~pos & hit).sum()), int((pos & ~hit).sum())
        tn = len(pos) - tp - fp - fn
        out['auc_' + c] = auc(pos, P[:, j])
        out['pr_auc_' + c] = average_precision(pos, P[:, j])
        out['sensitivity_' + c] = tp / (tp + fn) if tp + fn else np.nan
        out['specificity_' + c] = tn / (tn + fp) if tn + fp else np.nan
        out['f1_' + c] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        out['citl_' + c], out['slope_' + c] = recalibration(Y[:, j], P[:, j])
        out['observed_' + c], out['predicted_' + c] = float(Y[:, j].mean()), float(P[:, j].mean())
    for name in ('auc', 'pr_auc', 'sensitivity', 'specificity', 'f1'):
        out['macro_' + name] = float(np.nanmean([out['%s_%s' % (name, c)] for c in CLASSES]))
    return out


def _patient_samples(patients, B, seed=0):
    """B bootstrap samples of whole patients, as arrays of window indices."""
    uniq, inv = np.unique(patients, return_inverse=True)
    order = np.argsort(inv, kind='mergesort')
    starts = np.r_[0, np.cumsum(np.bincount(inv))]
    rng = np.random.default_rng(seed)
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([order[starts[i]:starts[i + 1]] for i in pick])


def metrics_table(Y, P, patients, B=1000, seed=0):
    """Every metric with its 95% interval (bootstrap over patients). Returns a DataFrame: metric, value, low, high."""
    import pandas as pd
    import warnings
    point = metrics(Y, P)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)              # a class absent from a resample: NaN, skipped
        boot = [metrics(Y[i], P[i]) for i in _patient_samples(patients, B, seed)]
    rows = []
    for k, v in point.items():
        b = np.array([d[k] for d in boot], np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lo, hi = np.nanpercentile(b, [2.5, 97.5]) if np.isfinite(b).any() else (np.nan, np.nan)
        rows.append(dict(metric=k, value=round(v, 4), ci95_lo=round(float(lo), 4), ci95_hi=round(float(hi), 4)))
    return pd.DataFrame(rows)


SUMMARY = ['kl', 'brier', 'macro_auc', 'macro_pr_auc', 'macro_f1', 'accuracy', 'sensitivity_seizure',
           'specificity_seizure', 'auc_seizure', 'slope_seizure', 'citl_seizure']


def summary_table(tables, metrics=SUMMARY):
    """{model: metrics_table} -> one row per model, 'value [low, high]' per metric."""
    import pandas as pd
    return pd.DataFrame({m: {r.metric: '%.3f [%.3f, %.3f]' % (r.value, r.ci95_lo, r.ci95_hi)
                             for r in t.itertuples() if r.metric in metrics} for m, t in tables.items()}).T[metrics]


def calibration_table(table):
    """Per class: observed and predicted mean share, calibration-in-the-large and slope, with intervals."""
    import pandas as pd
    t = table.set_index('metric')
    f = lambda k: '%.3f [%.3f, %.3f]' % tuple(t.loc[k, ['value', 'ci95_lo', 'ci95_hi']])
    return pd.DataFrame({c: {'observed share': '%.3f' % t.loc['observed_' + c, 'value'],
                             'mean predicted': '%.3f' % t.loc['predicted_' + c, 'value'],
                             'calibration-in-the-large': f('citl_' + c), 'calibration slope': f('slope_' + c)}
                         for c in CLASSES}).T


# ----------------------------------------------------------------------------- error analysis
def confusion(Y, P):
    """Counts: rows = majority class of the experts, columns = predicted class."""
    import pandas as pd
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    np.add.at(cm, (np.asarray(Y).argmax(1), np.asarray(P).argmax(1)), 1)
    return pd.DataFrame(cm, index=pd.Index(CLASSES, name='experts'), columns=pd.Index(CLASSES, name='model'))


def vote_entropy(Y):
    """Disagreement between experts: entropy of the vote shares (0 = unanimous; log 6 = uniform)."""
    Y = np.asarray(Y, np.float64)
    return -(Y * np.log(np.clip(Y, 1e-12, 1))).sum(1)


def errors_by(Y, P, groups, name):
    """Mean KL and window count per group (e.g. majority class, number of votes, disagreement band)."""
    import pandas as pd
    k = kl_per_sample(Y, P)
    t = pd.DataFrame({name: groups, 'kl': k}).groupby(name, observed=True).kl.agg(['mean', 'size'])
    return t.rename(columns={'mean': 'mean KL', 'size': 'windows'}).round(3)


def patient_concentration(Y, P, patients, top=(1, 5, 10)):
    """Share of the total error (sum of KL) carried by the patients with the largest total error,
    next to their share of the windows: are the errors spread out or concentrated in a few patients?"""
    import pandas as pd
    k = kl_per_sample(Y, P)
    t = pd.DataFrame({'p': patients, 'kl': k}).groupby('p').kl.agg(['sum', 'size']).sort_values('sum', ascending=False)
    return pd.DataFrame([dict(patients=n, share_of_error=round(float(t['sum'].head(n).sum() / t['sum'].sum()), 3),
                              share_of_windows=round(float(t['size'].head(n).sum() / t['size'].sum()), 3))
                         for n in top])


# ----------------------------------------------------------------------------- plots
def plot_calibration(Y, preds, bins=10, axes=None):
    """One panel per class: observed vote share against mean predicted probability, in bins of equal size.
    preds: {label: P}. The diagonal is perfect calibration."""
    import matplotlib.pyplot as plt
    from hms_eval import STYLE, _INK, _GRID
    if axes is None:
        _, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    colours = {v[0]: v[1] for v in STYLE.values()}
    for j, (ax, c) in enumerate(zip(np.ravel(axes), CLASSES)):
        ax.plot([0, 1], [0, 1], color=_GRID, lw=1.2)
        for label, P in preds.items():
            order = np.argsort(P[:, j])
            parts = np.array_split(order, bins)
            x = [P[i, j].mean() for i in parts]
            y = [Y[i, j].mean() for i in parts]
            ax.plot(x, y, marker='o', ms=4, lw=1.5, color=colours.get(label), label=label)
        ax.set_title(c if c in ('seizure', 'other') else c.upper(), fontsize=9, color=_INK, loc='left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('mean predicted probability', fontsize=8)
        ax.set_ylabel('observed share of votes', fontsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    np.ravel(axes)[0].legend(frameon=False, fontsize=7)
    plt.tight_layout()
    return axes


def plot_confusion(cm, ax=None, title=None):
    """Confusion matrix as row percentages (each row: where that class's windows went)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(5.2, 4.4))[1]
    pct = 100 * cm.values / np.maximum(cm.values.sum(1, keepdims=True), 1)
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
    names = [c.upper() if c != 'seizure' and c != 'other' else c for c in CLASSES]
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, '%d' % round(pct[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if pct[i, j] > 55 else '#1f1f1e')
    ax.set_xlabel('predicted class')
    ax.set_ylabel('majority class of the experts')
    ax.set_title(title or 'row %', fontsize=9, loc='left')
    return ax
