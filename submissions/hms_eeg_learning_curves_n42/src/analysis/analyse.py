import sys, os, json, glob
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the team folder
TRAIN_CSV = os.environ.get('HMS_TRAIN_CSV', 'train.csv')   # train.csv of the competition
sys.path.insert(0, ROOT + '/src')
import numpy as np, pandas as pd
import hms_data as hd, hms_eval as he
R = ROOT + '/results'
pd.set_option('display.width', 200)

# --- rebuild the split from train.csv, check it is the Kaggle one
meta = hd.select_windows(pd.read_csv(TRAIN_CSV), seed=0)
meta['total_votes'] = meta[hd.VOTE_COLS].sum(1).astype(int)
meta = hd.assign_split(meta)
hq = meta.total_votes >= hd.HQ_VOTES
te = meta[(meta.split == 'test') & (meta.is_primary | hq)]
ids, sub = np.load(R + '/preds/ids_test.npy'), np.load(R + '/preds/subids_test.npy')
print('test rows local %d vs kaggle %d' % (len(te), len(ids)))
print('same (eeg_id, sub_id) set:', set(zip(te.eeg_id, te.eeg_sub_id)) == set(zip(ids, sub)))

# --- describe split
print(pd.DataFrame(hd.describe_split(meta)).T)

# --- class mix: primary windows by vote group, and hq windows, per split
Y = hd.soft_targets(meta)
def mix(mask): return pd.Series(Y[mask.to_numpy()].mean(0), hd.CLASSES).round(3)
rows = {}
for s in ['train', 'val', 'test']:
    sp = meta.split == s
    rows[s + ' primary <10 votes'] = mix(sp & meta.is_primary & ~hq)
    rows[s + ' all hq windows'] = mix(sp & hq)
print(pd.DataFrame(rows).T)

# hq windows per patient, per split
for s in ['train', 'val', 'test']:
    v = meta[(meta.split == s) & hq]
    c = v.groupby('patient_id').size().sort_values(ascending=False)
    print(s, 'hq windows', len(v), 'patients', len(c), 'top5 share %.2f' % (c.head(5).sum() / len(v)),
          'top1', c.iloc[0], 'median', int(c.median()))
# primary windows of the train pool that are hq
tr = meta[(meta.split == 'train') & meta.is_primary]
print('train pool primary windows', len(tr), 'with >=10 votes', int((tr.total_votes >= 10).sum()))

# --- floors: training-vote mean (as run) vs mean of TRAINING hq windows, scored on test hq
Yte = np.load(R + '/preds/targets_test.npy'); hq_t = np.load(R + '/preds/hq_test.npy'); pr_t = np.load(R + '/preds/primary_test.npy')
pats = np.load(R + '/preds/patients_test.npy')
for name, m in [('mean of all train primary', meta.is_primary & (meta.split == 'train')),
                ('mean of train hq windows', hq & (meta.split == 'train'))]:
    q = Y[m.to_numpy()].mean(0)
    k = he.kl_per_sample(Yte, np.tile(q, (len(Yte), 1)))
    print('floor %-28s test hq %.3f  test all %.3f' % (name, k[hq_t].mean(), k[pr_t].mean()))

# --- seeds pooled paired test: mean KL over seeds per window, bootstrap over patients
d = he.load_results(R)
def kl_avg(m, n):
    s = sorted(d[(d.model == m) & (d.n_train == n)].seed)
    return np.mean([he.kl_per_sample(Yte, np.load('%s/preds/%s_n%d_s%d_test.npy' % (R, m, n, i))) for i in s], 0)
out = []
for a, b in [('cnn1d', 'minirocket'), ('cnn1d', 'lgbm'), ('minirocket', 'lgbm'), ('tabpfn', 'lgbm'), ('minirocket', 'tabpfn'), ('cnn1d', 'tabpfn')]:
    for n in sorted(d.n_train.unique()):
        if not ({a, b} <= set(d[d.n_train == n].model)): continue
        for which, mask in [('hq', hq_t), ('all', pr_t)]:
            diff = (kl_avg(a, n) - kl_avg(b, n))[mask]
            boot = he.cluster_bootstrap(diff, pats[mask], 4000, 1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append(dict(pair=a + ' - ' + b, N=n, test=which, delta=round(diff.mean(), 3), lo=round(lo, 3), hi=round(hi, 3)))
o = pd.DataFrame(out)
print(o[o.test == 'hq'].to_string(index=False))
print(o[o.test == 'all'].to_string(index=False))
o.to_csv(os.environ.get('OUT', '/dev/null'), index=False)

# --- per-patient: how much of the CNN-vs-LGBM gap at full N comes from few patients
n = d.n_train.max()
diff = (kl_avg('cnn1d', n) - kl_avg('lgbm', n))[hq_t]
pp = pd.DataFrame(dict(p=pats[hq_t], d=diff)).groupby('p').d.agg(['sum', 'size', 'mean']).sort_values('sum')
print('full N cnn-lgbm on hq: total %.3f; 5 most favourable patients give %.0f%% of the sum; patients with cnn better: %d/%d'
      % (diff.mean(), 100 * pp['sum'].head(5).sum() / pp['sum'].sum(), (pp['mean'] < 0).sum(), len(pp)))
print(pp.head(6)); print(pp.tail(4))

# --- CNN val curves at full N and 1000
for f in ['cnn1d_n14215_s0', 'cnn1d_n1000_s0', 'cnn1d_n100_s0']:
    i = json.load(open('%s/%s.json' % (R, f)))['info']
    print(f, 'val', i['val_kl_per_epoch'][:30]); print('   train', i['train_loss_per_epoch'][:30])
