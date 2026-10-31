import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the team folder
TRAIN_CSV = os.environ.get('HMS_TRAIN_CSV', 'train.csv')   # train.csv of the competition
sys.path.insert(0, ROOT + '/src')
import numpy as np, pandas as pd, hms_eval as he
R = ROOT + '/results'
L = lambda f: np.load('%s/preds/%s.npy' % (R, f))
Y, hq, ids, pats = L('targets_test'), L('hq_test'), L('ids_test'), L('patients_test')
d = he.load_results(R)
rows = []
for (m, n, s), g in d.groupby(['model', 'n_train', 'seed']):
    k = he.kl_per_sample(Y, L('%s_n%d_s%d_test' % (m, n, s)))[hq]
    rec = pd.Series(k).groupby(ids[hq]).mean()
    pat = pd.Series(k).groupby(pats[hq]).mean()
    rows.append(dict(model=m, N=n, seed=s, window=k.mean(), recording=rec.mean(), patient=pat.mean(),
                     no56885=k[pats[hq] != 56885].mean()))
t = pd.DataFrame(rows).groupby(['model', 'N'])[['window', 'recording', 'patient', 'no56885']].mean().round(3)
for c in t.columns:
    print('==', c); print(t[c].unstack(0).to_string())
