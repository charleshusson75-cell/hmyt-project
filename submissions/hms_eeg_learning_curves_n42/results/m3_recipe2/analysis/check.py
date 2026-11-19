# python check.py <this results folder (with preds/, not committed)> <src folder>
import sys
R, CODE = sys.argv[1], sys.argv[2]
sys.path.insert(0, CODE)
import numpy as np, pandas as pd, hms_eval as he
pd.set_option('display.width', 220)
pp = he.paired_pooled(R, [('cnn1d_ft', 'logreg_both'), ('cnn1d_ft', 'minirocket'), ('cnn1d', 'logreg_both')])
print(pp[pp.N >= 3000].to_string(index=False))
L = lambda f: np.load('%s/preds/%s.npy' % (R, f))
Y, hq, ids, pats, prim = L('targets_test'), L('hq_test'), L('ids_test'), L('patients_test'), L('primary_test')
big = pd.Series(pats[hq]).value_counts().index[0]
d = he.load_results(R); n = 14215
rows = {}
for m in ['lgbm', 'logreg_band', 'minirocket', 'logreg_both', 'cnn1d', 'cnn1d_ft']:
    out = []
    for s in sorted(d[(d.model == m) & (d.n_train == n)].seed):
        k = he.kl_per_sample(Y, L('%s_n%d_s%d_test' % (m, n, s)))[hq]
        out.append([k.mean(), pd.Series(k).groupby(ids[hq]).mean().mean(), pd.Series(k).groupby(pats[hq]).mean().mean(), k[pats[hq] != big].mean()])
    rows[m] = np.round(np.mean(out, 0), 3)
print('\nN = 14,215, test KL on >= 10-vote windows, averaged 4 ways (patient %s has the most windows)' % big)
print(pd.DataFrame(rows, index=['per window', 'per recording', 'per patient', 'without that patient']).T.to_string())
print('\nmean predicted class shares, N = 14,215 (seed 0), vs truth')
cls = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
tab = {'truth, >=10-vote windows': Y[hq].mean(0), 'truth, other primary windows': Y[prim & ~hq].mean(0)}
for m in ('cnn1d', 'cnn1d_ft'):
    P = L('%s_n%d_s0_test' % (m, n))
    tab[m + ', >=10-vote windows'] = P[hq].mean(0)
    tab[m + ', other primary windows'] = P[prim & ~hq].mean(0)
print(pd.DataFrame(tab, index=cls).T.round(3).to_string())
