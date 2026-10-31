"""Scoring and saving one run, and reading results back for the learning curves (protocol v2).

    res = score(split, val_proba, test_proba)
    save_result(out_dir, model_name, split, res, info, val_proba, test_proba)
    summarise(out_dir)  /  curve_table(out_dir)  /  plot_curve(out_dir)  /  paired_table(out_dir)

Two test sets (see hms_data): "hq" = every stored test window with >= 10 votes (reliable labels,
the PRIMARY metric, `test_kl_hq`), and "all" = one window per test recording (`test_kl_all`).
Several hq windows can come from one recording, so confidence intervals and paired tests resample
PATIENTS (cluster bootstrap), not windows. Two uncertainties are kept apart: the bootstrap interval
over test patients and, in summarise(), the SD across seeds.
"""
import os, json, glob, hashlib, datetime
import numpy as np
from hms_data import CLASSES, HQ_VOTES

PRIMARY = 'test_kl_hq'


def kl_per_sample(P, Q, eps=1e-15):
    """KL(p || q) per window, with 0 log 0 = 0 and q clipped at eps."""
    P = np.asarray(P, np.float64)
    Q = np.clip(np.asarray(Q, np.float64), eps, 1.0)
    logp = np.log(np.where(P > 0, P, 1.0))
    return np.where(P > 0, P * (logp - np.log(Q)), 0.0).sum(1)


def cluster_bootstrap(values, clusters, B=2000, seed=0):
    """Means of B resamples of whole clusters (patients) with replacement."""
    values, clusters = np.asarray(values, np.float64), np.asarray(clusters)
    _, inv = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv).astype(np.float64)
    C = len(sums)
    w = np.random.default_rng(seed).multinomial(C, np.full(C, 1.0 / C), size=B)
    return (w @ sums) / (w @ counts)


def kl_ci(P, Q, clusters, B=2000, alpha=0.05, seed=0):
    k = kl_per_sample(P, Q)
    lo, hi = np.percentile(cluster_bootstrap(k, clusters, B, seed), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(k.mean()), float(lo), float(hi)


def per_class_auc(Y, Q):
    """One-vs-rest AUC per class, against the majority (consensus) label."""
    from sklearn.metrics import roc_auc_score
    yc = np.asarray(Y).argmax(1)
    out = {}
    for j, c in enumerate(CLASSES):
        pos = yc == j
        out[c] = round(float(roc_auc_score(pos, Q[:, j])), 4) if 0 < pos.sum() < len(pos) else None
    return out


def _check(Q, n):
    Q = np.asarray(Q, np.float64)
    if Q.shape != (n, len(CLASSES)) or not np.isfinite(Q).all():
        raise ValueError('predictions must be a finite (%d, %d) array, got %s' % (n, len(CLASSES), Q.shape))
    if np.abs(Q.sum(1) - 1).max() > 1e-3:
        raise ValueError('prediction rows must sum to 1 (max error %.2e)' % np.abs(Q.sum(1) - 1).max())
    return Q


def score(split, val_p, test_p):
    val_p, test_p = _check(val_p, len(split.Y_val)), _check(test_p, len(split.Y_test))
    Yt, hq, prim, pats = split.Y_test, split.test_hq, split.test_primary, split.patients_test
    k = kl_per_sample(Yt, test_p)
    kl_hq, lo_hq, hi_hq = kl_ci(Yt[hq], test_p[hq], pats[hq])
    kl_all, lo_all, hi_all = kl_ci(Yt[prim], test_p[prim], pats[prim])
    prior = np.tile(split.Y_train.mean(0, dtype=np.float64), (len(Yt), 1))
    kf = kl_per_sample(Yt, prior)
    pc = per_class_auc(Yt[hq], test_p[hq])
    aucs = [v for v in pc.values() if v is not None]
    low = prim & ~hq
    kv = kl_per_sample(split.Y_val, val_p)
    return dict(
        n_train=int(split.n_train), n_patients_train=int(split.n_patients_train), seed=int(split.seed),
        test_kl_hq=round(kl_hq, 5), test_kl_hq_ci95=[round(lo_hq, 5), round(hi_hq, 5)],
        test_kl_all=round(kl_all, 5), test_kl_all_ci95=[round(lo_all, 5), round(hi_all, 5)],
        test_kl_all_lowvotes=round(float(k[low].mean()), 5) if low.any() else None,
        floor_test_kl_hq=round(float(kf[hq].mean()), 5), floor_test_kl_all=round(float(kf[prim].mean()), 5),
        beats_floor=bool(kl_hq < kf[hq].mean() - 1e-9),                    # a tie is not a win
        val_kl_hq=round(float(kv[split.val_hq].mean()), 5), val_kl_all=round(float(kv[split.val_primary].mean()), 5),
        test_auc_macro=round(float(np.mean(aucs)), 4) if aucs else None, test_auc_per_class=pc,
        test_acc_consensus=round(float((Yt[hq].argmax(1) == test_p[hq].argmax(1)).mean()), 4),
        n_test_hq=int(hq.sum()), n_test_all=int(prim.sum()), n_test_hq_patients=int(len(set(pats[hq]))),
        n_val=len(split.Y_val), n_test=len(Yt), primary_metric=PRIMARY,
        test_ids_hash=hashlib.md5(np.stack([split.ids_test, split.subids_test]).astype(np.int64).tobytes()).hexdigest()[:12])


def one_line(model, r):
    return ('%-10s n=%-6d seed %d | test KL hq %.4f [%.4f, %.4f] all %.4f | floor %.4f (%s) | AUC %s'
            % (model, r['n_train'], r['seed'], r['test_kl_hq'], *r['test_kl_hq_ci95'], r['test_kl_all'],
               r['floor_test_kl_hq'], 'beats it' if r['beats_floor'] else 'does NOT beat it',
               '%.3f' % r['test_auc_macro'] if r['test_auc_macro'] is not None else 'n/a'))


def save_result(out_dir, model, split, res, info, val_p, test_p):
    tag = '%s_n%d_s%d' % (model, split.n_train, split.seed)
    os.makedirs(os.path.join(out_dir, 'preds'), exist_ok=True)
    np.save(os.path.join(out_dir, 'preds', tag + '_val.npy'), np.asarray(val_p, np.float32))
    np.save(os.path.join(out_dir, 'preds', tag + '_test.npy'), np.asarray(test_p, np.float32))
    # the test windows, once per results folder, so window-by-window comparisons need nothing else
    for name, arr in (('targets_test', split.Y_test), ('ids_test', split.ids_test),
                      ('subids_test', split.subids_test), ('patients_test', split.patients_test),
                      ('total_votes_test', split.total_votes_test), ('primary_test', split.test_primary),
                      ('hq_test', split.test_hq)):
        f = os.path.join(out_dir, 'preds', name + '.npy')
        if not os.path.exists(f):
            np.save(f, np.asarray(arr))
        elif not np.array_equal(np.load(f), np.asarray(arr)):
            raise ValueError('this results folder already holds a different test set (%s)' % name)
    path = os.path.join(out_dir, tag + '.json')
    with open(path, 'w') as fh:
        json.dump(dict(model=model, tag=tag, time=datetime.datetime.now().isoformat(timespec='seconds'),
                       **res, info=info), fh, indent=1, default=float)
    return path


def load_results(out_dir):
    """Every run file (<model>_n<N>_s<seed>.json) of a results folder; other JSON files (tuning.json) are ignored."""
    import pandas as pd
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, '*_n*_s*.json')))]
    return pd.DataFrame([r for r in rows if 'model' in r and 'tag' in r])


def _metric(d, metric=None):
    """The column to plot: the primary metric of v2 results, test_kl for v1 results."""
    return metric or (PRIMARY if PRIMARY in d else 'test_kl')


def summarise(out_dir, metric=None):
    """One row per (model, N). Refuses to mix results scored on different test sets."""
    d = load_results(out_dir)
    if d.empty:
        return d
    if d.test_ids_hash.nunique() > 1:
        raise ValueError('results come from different test sets (%s) - not comparable'
                         % sorted(d.test_ids_hash.unique()))
    m = _metric(d, metric)
    # best epoch / epochs run: close to 1 means the model was still improving -> give it more epochs
    d['best_epoch_frac'] = [i['best_epoch'] / i['epochs'] if isinstance(i, dict) and i.get('epochs') else np.nan
                            for i in d['info']]
    agg = dict(seeds=('seed', 'nunique'), test_kl=(m, 'mean'), kl_sd_seeds=(m, 'std'),
               auc_macro=('test_auc_macro', 'mean'), acc=('test_acc_consensus', 'mean'),
               best_epoch_frac=('best_epoch_frac', 'max'))
    for extra in ('test_kl_all', 'floor_test_kl_hq', 'test_kl_all_lowvotes', 'beats_floor'):
        if extra in d:
            agg[extra] = (extra, 'all' if extra == 'beats_floor' else 'mean')
    return (d.groupby(['model', 'n_train']).agg(**agg).reset_index()
             .sort_values(['n_train', 'test_kl']).reset_index(drop=True))


# ------------------------------------------------------------------------------------- plots
# one fixed colour and marker per model, whatever the subset plotted (colour-blind-safe order);
# the line style says what the model reads: raw signal, spectrogram, or band-power features
STYLE = {'lgbm': ('LightGBM', '#2a78d6', 'o', 'features'),
         'cnn1d': ('1D CNN (ResNet)', '#eb6834', 's', 'signal'),
         'xresnet1d': ('CNN (xresnet1d, v1)', '#eb6834', 's', 'signal'),
         'minirocket': ('MiniRocket', '#1baf7a', 'D', 'signal'),
         'tabpfn': ('TabPFN', '#eda100', '^', 'features')}
INPUT_STYLE = {'signal': ('-', 'raw signal'), 'spectrogram': ('--', 'spectrogram'),
               'features': (':', 'band-power features')}
_EXTRA = ['#e87ba4', '#008300', '#6250d6', '#e34948']          # next models, in order of appearance
_INK, _MUTED, _GRID = '#1f1f1e', '#6b6a63', '#e6e5e0'


def _models(d):
    return [m for m in STYLE if m in set(d.model)] + sorted(set(d.model) - set(STYLE) - {'floor'})


def _style(models):
    st, extra = {}, iter(_EXTRA)
    for m in models:
        st[m] = STYLE.get(m) or (m, next(extra, _MUTED), 'v', 'signal')
    return st


def curve_table(out_dir, metric=None):
    """Test KL per (N, model): 'mean (SD over seeds)', one column per model, baseline first."""
    import pandas as pd
    d = load_results(out_dir)
    g = d.groupby(['n_train', 'model'])[_metric(d, metric)]
    mu, sd = g.mean().unstack(), g.std().unstack()
    cols = (['floor'] if 'floor' in mu else []) + [m for m in mu if m != 'floor']
    return pd.DataFrame({m: ['%.3f (%.3f)' % (mu.loc[n, m], sd.loc[n, m]) if mu.loc[n, m] == mu.loc[n, m] else '-'
                             for n in mu.index] for m in cols}, index=mu.index.rename('N'))


def _spread(y, gap):
    """Label heights moved apart to at least `gap`; labels that collide are spread around their mean."""
    y = np.asarray(y, float)
    groups = [[i] for i in np.argsort(y)]
    bottom = lambda g: y[g].mean() - gap * (len(g) - 1) / 2
    k = 0
    while k < len(groups) - 1:
        if bottom(groups[k + 1]) - (bottom(groups[k]) + gap * (len(groups[k]) - 1)) < gap:
            groups[k] += groups.pop(k + 1)
            k = 0
        else:
            k += 1
    out = np.empty_like(y)
    for g in groups:
        out[g] = bottom(g) + gap * np.arange(len(g))
    return out


def plot_curve(out_dir, ax=None, metric=None):
    """Learning curves: mean test KL over seeds, band = ±1 SD, the baseline dashed grey, colour = model,
    line style = input (raw signal / spectrogram / band power)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = load_results(out_dir)
    m = _metric(d, metric)
    ax = ax or plt.subplots(figsize=(7.6, 4.8))[1]
    ax.grid(True, axis='y', color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ends = []                                           # end-of-line labels: (x, y, text, colour)
    fl = d[d.model == 'floor'].groupby('n_train')[m].mean()
    if len(fl):
        ax.plot(fl.index, fl.values, color=_MUTED, lw=1.3, ls=(0, (6, 3)))
        ends.append((fl.index[-1] * 1.08, fl.values[-1], 'mean-vote baseline', _MUTED))
    st, used = _style(_models(d)), set()
    for mod, (label, c, mk, inp) in st.items():
        g = d[d.model == mod].groupby('n_train')[m]
        mu, sd = g.mean(), g.std().fillna(0)
        ls = INPUT_STYLE[inp][0]
        used.add(inp)
        ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.13, lw=0)
        ax.plot(mu.index, mu.values, color=c, lw=2, ls=ls, marker=mk, ms=6, mec='white', mew=1.2, label=label)
        ends.append((mu.index[-1] * 1.08, mu.values[-1], label, _INK))
    lo, hi = ax.get_ylim()
    for (x, _, text, c), y in zip(ends, _spread([e[1] for e in ends], 0.04 * (hi - lo))):
        ax.text(x, y, text, color=c, va='center', fontsize=8)
    ns = sorted(d.n_train.unique())
    labels = ['%gk' % round(n / 1000, 1) if n >= 1000 else str(n) for n in ns]
    if len(ns) > 1 and ns[-1] / ns[-2] < 1.6:          # e.g. 10k and 13.7k: second line, no overlap
        labels[-1] = '\n' + labels[-1]
    ax.set_xscale('log')
    ax.set_xlim(ns[0] * 0.8, ns[-1] * 3.5)
    ax.set_xticks(ns)
    ax.set_xticklabels(labels)
    ax.minorticks_off()
    ax.set_xlabel('training recordings N (log scale)')
    ax.set_ylabel('test KL divergence (lower is better)')
    ax.set_title('%s: mean over seeds, band = ±1 SD' % ('high-quality test windows (>= 10 votes)'
                 if m == PRIMARY else m), fontsize=9, color=_INK, loc='left')
    leg1 = ax.legend(frameon=False, fontsize=8, loc='lower left', title='model', title_fontsize=8)
    ax.add_artist(leg1)
    proxies = [Line2D([], [], color=_INK, lw=1.6, ls=INPUT_STYLE[i][0]) for i in INPUT_STYLE if i in used]
    ax.legend(proxies, [INPUT_STYLE[i][1] for i in INPUT_STYLE if i in used], frameon=False, fontsize=8,
              loc='upper right', title='input', title_fontsize=8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return ax


def plot_per_class_auc(out_dir, sizes=(1000, 10000)):
    """One-vs-rest AUC of each class (majority label, high-quality windows), one dot per model, one panel per N."""
    import pandas as pd, matplotlib.pyplot as plt
    d = load_results(out_dir)
    sizes = [n for n in sizes if n in set(d.n_train)] or sorted(d.n_train.unique())[-2:]   # else the 2 largest N
    models = _models(d)
    st = _style(models)
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.7 * len(sizes), 3.4), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], sizes):
        ax.grid(True, axis='x', color=_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for j, mod in enumerate(models):
            g = d[(d.model == mod) & (d.n_train == n)]
            if g.empty:
                continue
            auc = pd.DataFrame(list(g.test_auc_per_class)).mean()[CLASSES]
            label, c, mk, _ = st[mod]
            ax.plot(auc.values, np.arange(len(CLASSES)) + (j - (len(models) - 1) / 2) * 0.15, ls='none',
                    marker=mk, ms=6, color=c, mec='white', mew=0.8, label=label)
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels([c.upper() if c not in ('seizure', 'other') else c for c in CLASSES])
        ax.invert_yaxis()
        ax.set_xlim(0.55, 1.0)
        ax.set_xlabel('AUC, one class vs rest')
        ax.set_title('N = {:,}'.format(n), fontsize=9, color=_INK, loc='left')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=7.5, loc='lower right')
    fig.tight_layout()
    return fig


def _test_arrays(out_dir):
    P = lambda name: np.load(os.path.join(out_dir, 'preds', name + '.npy'))
    if os.path.exists(os.path.join(out_dir, 'preds', 'hq_test.npy')):
        return P('targets_test'), P('patients_test'), P('hq_test'), P('primary_test'), P('total_votes_test')
    Y = P('targets_test')                            # v1 folders: one window per recording, no patients
    ones = np.ones(len(Y), bool)
    return Y, np.arange(len(Y)), P('total_votes_test') >= HQ_VOTES, ones, P('total_votes_test')


def kl_by_votes(out_dir, n):
    """Mean test KL by number of expert votes, one window per test recording, every model at size n."""
    import pandas as pd
    d = load_results(out_dir)
    Y, _, _, prim, tv = _test_arrays(out_dir)
    grp = pd.cut(tv[prim], [0, 1, 2, 5, 9, 1000], labels=['1', '2', '3-5', '6-9', '>=10'])
    rows = {}
    for mod, g in d[d.n_train == n].groupby('model'):
        k = np.mean([kl_per_sample(Y, np.load(os.path.join(out_dir, 'preds', '%s_n%d_s%d_test.npy' % (mod, n, s))))
                     for s in g.seed], 0)[prim]
        rows[mod] = pd.Series(k).groupby(grp, observed=True).mean()
    out = pd.DataFrame(rows).T.round(3)
    out.loc['test windows'] = pd.Series(grp).value_counts().reindex(out.columns).values
    return out


def paired(out_dir, a, b, which='hq', B=2000, seed=0):
    """Model a minus model b on the same test windows ('hq' or 'all'), for every (N, seed) both have.
    Negative delta = a is better. Interval and p-value: bootstrap over test patients."""
    import pandas as pd
    Y, pats, hq, prim, _ = _test_arrays(out_dir)
    mask = hq if which == 'hq' else prim
    d = load_results(out_dir)
    rows = []
    for (n, s), g in d.groupby(['n_train', 'seed']):
        if not {a, b} <= set(g.model):
            continue
        k = {m: kl_per_sample(Y, np.load(os.path.join(out_dir, 'preds', '%s_n%d_s%d_test.npy' % (m, n, s))))
             for m in (a, b)}
        diff = (k[a] - k[b])[mask]
        boot = cluster_bootstrap(diff, pats[mask], B, seed + int(n) + 7 * int(s))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(n_train=int(n), seed=int(s), delta=round(float(diff.mean()), 4),
                         ci95=[round(float(lo), 4), round(float(hi), 4)],
                         p=round(float(min(1.0, 2 * min((boot >= 0).mean(), (boot <= 0).mean()))), 4)))
    return pd.DataFrame(rows)


def paired_table(out_dir, pairs=None, which='hq', B=2000):
    """Every pair of models, every N: mean delta over seeds (negative = first model better) and in how many
    seeds each model is significantly better (bootstrap over patients, p < 0.05). Baseline pairs last."""
    import itertools, pandas as pd
    d = load_results(out_dir)
    models = _models(d)
    if pairs is None:
        pairs = list(itertools.combinations(models, 2)) + [(m, 'floor') for m in models if 'floor' in set(d.model)]
    rows = []
    for a, b in pairs:
        p = paired(out_dir, a, b, which=which, B=B)
        for n, g in (p.groupby('n_train') if len(p) else []):
            rows.append(dict(pair='%s - %s' % (a, b), N=int(n), mean_delta=round(float(g.delta.mean()), 4),
                             first_better=int(((g.p < 0.05) & (g.delta < 0)).sum()),
                             second_better=int(((g.p < 0.05) & (g.delta > 0)).sum()), seeds=len(g)))
    return pd.DataFrame(rows)
