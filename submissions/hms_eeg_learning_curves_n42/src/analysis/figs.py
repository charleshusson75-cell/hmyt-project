"""Two extra figures for the Week 7 deck, drawn from the saved v2 predictions (and train.csv for the class mix)."""
import sys, json
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the team folder
TRAIN_CSV = os.environ.get('HMS_TRAIN_CSV', 'train.csv')   # train.csv of the competition
sys.path.insert(0, ROOT + '/src')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, hms_eval as he
R = ROOT + '/results'
INK, MUTED, GRID = he._INK, he._MUTED, he._GRID
L = lambda f: np.load('%s/preds/%s.npy' % (R, f))
Y, hq, ids, pats = L('targets_test'), L('hq_test'), L('ids_test'), L('patients_test')
d = he.load_results(R)
n = int(d.n_train.max())
big = pd.Series(pats[hq]).value_counts().index[0]            # patient with the most >=10-vote test windows

# --- 1. same predictions, four weightings, full N
views = ['per window\n(main metric)', 'per recording', 'per patient', 'without the patient\nwith most windows']
vals = {}
for m in ['lgbm', 'minirocket', 'cnn1d']:
    out = []
    for s in sorted(d[(d.model == m) & (d.n_train == n)].seed):
        k = he.kl_per_sample(Y, L('%s_n%d_s%d_test' % (m, n, s)))[hq]
        out.append([k.mean(), pd.Series(k).groupby(ids[hq]).mean().mean(), pd.Series(k).groupby(pats[hq]).mean().mean(),
                    k[pats[hq] != big].mean()])
    vals[m] = np.mean(out, 0)
fig, ax = plt.subplots(figsize=(7.2, 4.9))
w = 0.26
for j, m in enumerate(vals):
    label, col = he.STYLE[m][0], he.STYLE[m][1]
    x = np.arange(4) + (j - 1) * w
    ax.bar(x, vals[m], w * 0.92, color=col, label=label)
    for xi, v in zip(x, vals[m]):
        ax.text(xi, v + 0.012, '%.3f' % v, ha='center', va='bottom', fontsize=8, color=INK)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(views, fontsize=9)
ax.set_ylim(0, 0.95)
ax.set_ylabel('test KL, windows with >= 10 votes (lower is better)', fontsize=9)
ax.grid(True, axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.09))
ax.set_title('N = %s training recordings, mean over 3 seeds' % format(n, ','), fontsize=9, color=INK, loc='left', pad=22)
fig.savefig(R + '/figures/robustness_full_N.png', dpi=200, bbox_inches='tight'); plt.close(fig)
print('robustness', {m: np.round(v, 3).tolist() for m, v in vals.items()}, 'big patient', big)

# --- 2. class mix: what the models train on vs what they are scored on
mix = json.load(open(__import__('os').path.dirname(__file__) + '/mix.json'))
rows = [('training windows\n(what the models see)', 'training windows (all votes)'),
        ('training windows\nwith >= 10 votes', 'training windows, >=10 votes'),
        ('test windows with\n>= 10 votes (scored)', 'test windows, >=10 votes')]
names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
cols = ['#c44e52', '#8172b2', '#64b5cd', '#ccb974', '#55a868', '#b8b6b0']
fig, ax = plt.subplots(figsize=(6.3, 4.0))
for i, (lab, key) in enumerate(rows):
    left = 0.0
    for j, v in enumerate(mix['mix'][key]):
        ax.barh(i, v, 0.62, left=left, color=cols[j], edgecolor='white', lw=1, label=names[j] if i == 0 else None)
        if v >= 0.06 or j == 0:
            ax.text(left + v / 2, i, '%d%%' % round(100 * v), ha='center', va='center', fontsize=8.5,
                    color='white' if j in (0, 1, 4) else INK)
        left += v
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'], fontsize=8.5)
ax.set_xlabel('mean share of the expert votes', fontsize=9)
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.legend(frameon=False, fontsize=8.5, ncol=6, loc='upper center', bbox_to_anchor=(0.42, 1.12), handlelength=1, columnspacing=0.9)
fig.savefig(R + '/figures/class_mix_train_vs_test.png', dpi=200, bbox_inches='tight'); plt.close(fig)
print('ok')
