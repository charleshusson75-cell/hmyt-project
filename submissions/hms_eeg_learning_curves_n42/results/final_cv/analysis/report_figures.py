"""Figures of the final report (paper and slides), from the cross-validation results and report_numbers.py's tables.

    python report_figures.py <results folder> <code folder>

Writes PNGs (300 dpi) to figures/ next to this script. Colours: one per model, validated for colour-vision
deficiency (dataviz validator, all pairs within each panel): the learning curves are split into a band-power
panel and a raw-signal panel because no 7-colour set passes when every pair shares one chart.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT, CODE = sys.argv[1], sys.argv[2]
sys.path.insert(0, CODE)
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
INK, MUTED, GRID, SURF = '#1f1f1e', '#6b6a63', '#e6e5e0', '#ffffff'
NAME = {'floor': 'Mean-vote baseline', 'logreg_band': 'LR, band power', 'lgbm': 'LightGBM', 'tabpfn': 'TabPFN',
        'minirocket': 'MiniRocket', 'logreg_both': 'LR, MiniRocket + band power', 'cnn1d': '1D CNN',
        'cnn1d_ft': '1D CNN + fine-tuning'}
COL = {'lgbm': '#2a78d6', 'tabpfn': '#eda100', 'logreg_band': '#008300',            # band-power panel
       'minirocket': '#1baf7a', 'logreg_both': '#6250d6', 'cnn1d': '#eb6834', 'cnn1d_ft': '#a33a0e'}
MARK = {'lgbm': 'o', 'tabpfn': '^', 'logreg_band': 'v', 'minirocket': 'D', 'logreg_both': 'P', 'cnn1d': 's', 'cnn1d_ft': 's'}
SIZES = [100, 300, 1000, 3000, 10000, 'all']
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlesize': 8.5, 'axes.titlecolor': INK, 'legend.frameon': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'figure.facecolor': SURF, 'axes.facecolor': SURF})
kt = pd.read_csv(os.path.join(HERE, 'kl_table.csv'))
pool = 13671                                                # mean training pool over the 10 folds ('all')
X = {100: 100, 300: 300, 1000: 1000, 3000: 3000, 10000: 10000, 'all': pool}


def _x(n):
    return X[int(n)] if str(n) != 'all' else pool


def _clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _xaxis(ax):
    ax.set_xscale('log')
    ax.set_xticks([100, 300, 1000, 3000, 10000])
    ax.set_xticklabels(['100', '300', '1k', '3k', '10k'])
    ax.minorticks_off()
    ax.set_xlim(75, 22000)
    ax.set_xlabel('training recordings N (log scale; last point = all, ~13.7k)')


# ------------------------------------------------------------------ 1. learning curves, two panels
def curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True)
    groups = [('Band-power features (160 per window)', ['logreg_band', 'lgbm', 'tabpfn']),
              ('Raw signal (16 chains x 2,000 samples)', ['minirocket', 'logreg_both', 'cnn1d', 'cnn1d_ft'])]
    fl = kt[kt.model == 'floor']
    for ax, (title, ms) in zip(axes, groups):
        ax.plot([_x(n) for n in fl.N], fl.kl, color=MUTED, ls='--', lw=1.1)
        ax.text(2200, fl.kl.iloc[2] - 0.015, 'mean-vote baseline', color=MUTED, fontsize=6.5, va='top', ha='center')
        ends = []
        for m in ms:
            t = kt[kt.model == m]
            x = [_x(n) for n in t.N]
            ax.fill_between(x, t.lo, t.hi, color=COL[m], alpha=0.13, lw=0)
            ls = '-' if m != 'logreg_band' else '-'
            ax.plot(x, t.kl, color=COL[m], lw=1.6, ls=ls, marker=MARK[m], ms=4.2, mec=SURF, mew=0.6, label=NAME[m])
            nine = t.folds < 10
            if nine.any():                                     # cells pooled over 9 folds: open marker
                ax.plot(np.array(x)[nine], t.kl[nine], ls='none', marker=MARK[m], ms=4.2, mfc=SURF, mec=COL[m], mew=1.1)
        ax.set_title(title, loc='left')
        _xaxis(ax)
        _clean(ax)
        ax.legend(loc='upper right', fontsize=6.8, handlelength=2.2)
    axes[0].set_ylabel('test KL divergence, windows with >= 10 votes\n(lower is better)')
    axes[0].set_ylim(0.5, 1.33)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(FIG, 'fig_curves.png'))
    plt.close(fig)


# ------------------------------------------------------------------ 2. paired differences
def paired():
    nb = pd.read_csv(os.path.join(ROOT, 'figures', 'cv_paired.csv'))
    ex = pd.read_csv(os.path.join(HERE, 'paired_extra.csv'))
    allp = pd.concat([nb, ex]).drop_duplicates(['pair', 'N'])
    panels = [('(a) Input, same learner (LR):\nminus LR on MiniRocket features',
               [('logreg_band - minirocket', 'band power', '-', 'v'), ('logreg_both - minirocket', 'MiniRocket + band power', '--', 'P')]),
              ('(b) Fine-tuning on >= 10-vote windows:\nCNN + fine-tuning minus CNN',
               [('cnn1d_ft - cnn1d', 'CNN + fine-tuning', '-', 's')]),
              ('(c) CNN minus the best feature model\n(LR, MiniRocket + band power)',
               [('cnn1d_ft - logreg_both', 'CNN + fine-tuning', '-', 's'), ('cnn1d - logreg_both', 'CNN, no fine-tuning', '--', 'o')])]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), sharey=False)
    for ax, (title, rows) in zip(axes, panels):
        ax.axhline(0, color=MUTED, lw=0.9)
        for k, (pair, label, ls, mk) in enumerate(rows):
            t = allp[allp.pair == pair].copy()
            t['x'] = [_x(n) * (1.06 if k else 0.94) for n in t.N]
            col = INK if k == 0 else MUTED
            ax.errorbar(t.x, t.delta, yerr=[t.delta - t.ci95_lo, t.ci95_hi - t.delta], color=col, ls=ls, lw=1.3,
                        marker=mk, ms=4, mfc=col, mec=SURF, capsize=2, elinewidth=0.9, label=label)
        ax.set_title(title, loc='left', fontsize=7.5)
        _xaxis(ax)
        ax.set_xlabel('training recordings N')
        _clean(ax)
        ax.legend(fontsize=6.3, loc='upper left' if title.startswith('(a)') else 'upper right')
        if title.startswith('(a)'):
            ax.set_ylim(-0.11, 0.27)
    axes[0].set_ylabel('difference in test KL (95% CI)\n< 0: first model better')
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIG, 'fig_paired.png'))
    plt.close(fig)


# ------------------------------------------------------------------ 3. fold-to-fold spread
def folds():
    bf = pd.read_csv(os.path.join(ROOT, 'figures', 'cv_kl_by_fold.csv'))
    sd = pd.read_csv(os.path.join(HERE, 'seeds_fold0.csv'))
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for m, dx in (('logreg_both', -0.12), ('cnn1d_ft', 0.12)):
        ax.plot(bf['test fold'] + dx, bf[m], ls='none', marker=MARK[m], ms=5, color=COL[m], mec=SURF, mew=0.6, label=NAME[m])
    for _, r in bf.iterrows():
        if not np.isnan(r['cnn1d_ft']):
            ax.plot([r['test fold'] - 0.12, r['test fold'] + 0.12], [r['logreg_both'], r['cnn1d_ft']], color=GRID, lw=1, zorder=0)
    s = sd[(sd.model == 'cnn1d_ft') & (sd.n == sd[sd.model == 'cnn1d_ft'].n.max())].iloc[0]
    ax.errorbar([-1.1], [s['mean']], yerr=[s['std']], color=COL['cnn1d_ft'], capsize=2.5, lw=1.1, marker='_', ms=6)
    ax.text(-1.1, s['mean'] - s['std'] - 0.008, 'fold 0,\n3 seeds\n(+/- SD)', fontsize=5.6, color=MUTED, ha='center', va='top')
    ax.set_xticks(range(10))
    ax.set_xlim(-2.0, 9.5)
    ax.set_ylim(0.38, 0.9)
    ax.set_xlabel('test fold (about 100 patients each)')
    ax.set_ylabel('test KL, all training recordings')
    _clean(ax)
    ax.legend(fontsize=6.5, loc='upper left')
    ax.text(1, 0.57, 'no CNN\nrun', fontsize=5.6, color=MUTED, ha='center', va='top')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_folds.png'))
    plt.close(fig)


# ------------------------------------------------------------------ 4. calibration (common windows)
def calibration():
    import hms_eval as E
    from hms_data import CLASSES
    c = E.cv_collect(ROOT)
    ms = ['lgbm', 'minirocket', 'cnn1d', 'cnn1d_ft']
    ok = c.hq.copy()
    for m in ms:
        ok &= ~np.isnan(c.P[(m, 'all')][:, 0])
    Y = c.Y[ok]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.3))
    for j, (ax, cl) in enumerate(zip(axes.ravel(), CLASSES)):
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2)
        for m in ms:
            P = c.P[(m, 'all')][ok]
            parts = np.array_split(np.argsort(P[:, j]), 10)
            ax.plot([P[i, j].mean() for i in parts], [Y[i, j].mean() for i in parts], color=COL[m], lw=1.4,
                    marker=MARK[m], ms=3.5, mec=SURF, mew=0.5, label=NAME[m])
        ax.set_title(cl if cl in ('seizure', 'other') else cl.upper(), loc='left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _clean(ax)
        if j >= 3:
            ax.set_xlabel('mean predicted probability (deciles)')
        if j % 3 == 0:
            ax.set_ylabel('observed share of votes')
    axes[0, 0].legend(fontsize=6.3, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_calibration.png'))
    plt.close(fig)
    with open(os.path.join(HERE, 'fig_calibration_info.json'), 'w') as fh:
        json.dump(dict(windows=int(ok.sum()), patients=len(set(c.patients[ok]))), fh)


# ------------------------------------------------------------------ 5. confusion matrix
def confusion():
    cm = pd.read_csv(os.path.join(ROOT, 'figures', 'm4_confusion_cnn1d_ft.csv'), index_col=0)
    labs = ['seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'other']
    v = cm.values.astype(float)
    pct = 100 * v / v.sum(1, keepdims=True)
    blues = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#6da7ec', '#2a78d6', '#184f95'])
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.imshow(pct, cmap=blues, vmin=0, vmax=100)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, '%.0f' % pct[i, j], ha='center', va='center', fontsize=7,
                    color='white' if pct[i, j] > 55 else INK)
    ax.set_xticks(range(6))
    ax.set_xticklabels(labs, rotation=35, ha='right')
    ax.set_yticks(range(6))
    ax.set_yticklabels(['%s (%d)' % (l, n) for l, n in zip(labs, v.sum(1).astype(int))])
    ax.set_xlabel("model's most likely class")
    ax.set_ylabel("experts' majority class (windows)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_confusion.png'))
    plt.close(fig)


# ------------------------------------------------------------------ 6. explainability, global
def xai_global():
    shap = pd.read_csv(os.path.join(ROOT, 'figures', 'xai_lgbm_shap_by_band.csv'), index_col=0)
    occ = pd.read_csv(os.path.join(ROOT, 'figures', 'xai_cnn_occlusion.csv'), index_col=0)
    labs = ['seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'other']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1, 2.6]))
    greens = LinearSegmentedColormap.from_list('g', ['#ffffff', '#cde2fb', '#6da7ec', '#2a78d6', '#184f95'])
    a.imshow(shap.values, cmap=greens, vmin=0, vmax=0.75, aspect='auto')
    for i in range(6):
        for j in range(5):
            a.text(j, i, '%.0f' % (100 * shap.values[i, j]), ha='center', va='center', fontsize=6.8,
                   color='white' if shap.values[i, j] > 0.45 else INK)
    a.set_xticks(range(5))
    a.set_xticklabels(shap.columns, rotation=35, ha='right')
    a.set_yticks(range(6))
    a.set_yticklabels(labs)
    a.set_title('(a) LightGBM: % of |SHAP| per band', loc='left', fontsize=7.2)
    div = LinearSegmentedColormap.from_list('d', ['#2a78d6', '#9ec5f4', '#f2f1ee', '#f5b38f', '#eb6834', '#a33a0e'])
    lim = 0.1
    b.imshow(occ.values, cmap=div, vmin=-lim, vmax=lim, aspect='auto')
    for i in range(6):
        for j in range(16):
            val = occ.values[i, j]
            if abs(val) >= 0.02:
                b.text(j, i, '%.2f' % val, ha='center', va='center', fontsize=5.4, color='white' if abs(val) > 0.07 else INK)
    b.set_xticks(range(16))
    b.set_xticklabels(occ.columns, rotation=60, ha='right', fontsize=6.3)
    b.set_yticks(range(6))
    b.set_yticklabels(labs)
    for x in (3.5, 7.5, 11.5):
        b.axvline(x, color='white', lw=2)
    b.set_title("(b) CNN + fine-tuning: drop in P(experts' class) when one chain is removed\n"
                "orange: relied on; blue: removing it helps; |value| < 0.02 not printed", loc='left', fontsize=7.2)
    for ax in (a, b):
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_xai_global.png'))
    plt.close(fig)


# ------------------------------------------------------------------ 7. Grad-CAM cases (crops of the notebook's figures)
def gradcam_cases():
    from PIL import Image
    ok = Image.open(os.path.join(ROOT, 'figures', 'xai_cnn_gradcam.png')).convert('RGB')
    er = Image.open(os.path.join(ROOT, 'figures', 'xai_cnn_gradcam_errors.png')).convert('RGB')

    def cell(im, r, c):
        W, H = im.size
        return im.crop((c * W // 2, r * H // 3, (c + 1) * W // 2, (r + 1) * H // 3))
    left = [cell(ok, 0, 0), cell(ok, 1, 0), cell(ok, 0, 1)]           # seizure, GPD, LPD: correct
    right = [cell(er, 0, 0), cell(er, 0, 1), cell(er, 1, 1)]          # missed seizure, false seizure, GRDA -> other
    tags = [['A', 'B', 'C'], ['D', 'E', 'F']]
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.9))
    for k in range(3):
        for c, im in enumerate((left[k], right[k])):
            ax = axes[k, c]
            ax.imshow(np.asarray(im))
            ax.axis('off')
            ax.text(-0.01, 1.0, tags[c][k], transform=ax.transAxes, fontsize=9, fontweight='bold', ha='right', va='top')
    axes[0, 0].set_title('Correct (experts and model agree)', loc='left', fontsize=8.5, pad=6)
    axes[0, 1].set_title('Misclassified (experts and model disagree)', loc='left', fontsize=8.5, pad=6)
    fig.tight_layout(h_pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(FIG, 'fig_gradcam_cases.png'))
    plt.close(fig)


def gradcam_wide():
    """Slide versions: 2 rows x 3 panels (wider panels than the paper's 3 x 2)."""
    from PIL import Image
    ok = Image.open(os.path.join(ROOT, 'figures', 'xai_cnn_gradcam.png')).convert('RGB')
    er = Image.open(os.path.join(ROOT, 'figures', 'xai_cnn_gradcam_errors.png')).convert('RGB')

    def cell(im, r, c):
        W, H = im.size
        return im.crop((c * W // 2, r * H // 3, (c + 1) * W // 2, (r + 1) * H // 3))
    sets = {'fig_gradcam_cases_wide.png': ([cell(ok, 0, 0), cell(ok, 1, 0), cell(ok, 0, 1)],
                                           [cell(er, 0, 0), cell(er, 0, 1), cell(er, 1, 1)],
                                           ('Correct', 'Misclassified (Grad-CAM for the class the model chose)'), 'ABCDEF'),
            'fig_gradcam_errors_wide.png': ([cell(er, 0, 0), cell(er, 0, 1), cell(er, 1, 0)],
                                            [cell(er, 1, 1), cell(er, 2, 0), cell(er, 2, 1)], None, None)}
    for name, (top, bottom, heads, tags) in sets.items():
        fig, axes = plt.subplots(2, 3, figsize=(13.0, 5.6))
        for r, row in enumerate((top, bottom)):
            for c, im in enumerate(row):
                ax = axes[r, c]
                ax.imshow(np.asarray(im))
                ax.axis('off')
                if tags:
                    ax.text(-0.01, 1.0, tags[3 * r + c], transform=ax.transAxes, fontsize=12, fontweight='bold',
                            ha='right', va='top')
        if heads:
            axes[0, 0].set_title(heads[0], loc='left', fontsize=11, pad=4)
            axes[1, 0].set_title(heads[1], loc='left', fontsize=11, pad=4)
        fig.tight_layout(h_pad=0.6, w_pad=0.6)
        fig.savefig(os.path.join(FIG, name), dpi=220)
        plt.close(fig)


if __name__ == '__main__':
    for f in (curves, paired, folds, calibration, confusion, xai_global, gradcam_cases, gradcam_wide):
        f()
        print('done', f.__name__, flush=True)
