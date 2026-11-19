"""Numbers for the final report, from the cross-validation results (10 patient folds, seed 0).

    python report_numbers.py <results folder> <code folder>

Writes CSV tables next to this script: everything the paper and the slides quote comes from here.
- kl_table.csv            pooled test KL (>= 10 votes) per model and N, 95% interval over patients, folds used
- kl_table_common.csv     the same on the windows every model has (fair when a fold is missing for one model)
- paired_extra.csv        paired differences not in the notebook (CNN + fine-tuning vs the feature models, ...)
- best_by_N.csv           best model at each N and its paired difference with the runner-up
- first_beats_floor.csv   smallest N at which each model is significantly better than the mean-vote baseline
- fold_wins.csv           in how many folds each model is the best, per N
- m4_common.csv           Milestone 4 metrics, all training recordings, on the windows every model has
- seeds_fold0.csv         seed-to-seed spread on fold 0 (3 seeds), to compare with the fold-to-fold spread
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT, CODE = sys.argv[1], sys.argv[2]
sys.path.insert(0, CODE)
import hms_eval as E, hms_metrics as HM

MODELS = ['floor', 'logreg_band', 'lgbm', 'tabpfn', 'minirocket', 'logreg_both', 'cnn1d', 'cnn1d_ft']
B = 2000
c = E.cv_collect(ROOT)
Y, pats, hq = c.Y, c.patients, c.hq
out = lambda name: os.path.join(HERE, name)


def kl_row(m, size, mask):
    P = c.P[(m, size)]
    ok = mask & ~np.isnan(P[:, 0])
    k = E.kl_per_sample(Y[ok], P[ok])
    lo, hi = np.percentile(E.cluster_bootstrap(k, pats[ok], B, 0), [2.5, 97.5])
    return dict(model=m, N=size, kl=round(k.mean(), 4), lo=round(lo, 4), hi=round(hi, 4),
                folds=len(set(c.fold[ok])), windows=int(ok.sum()), patients=len(set(pats[ok])))


rows, rows_c = [], []
for size in E.SIZES:
    ms = [m for m in MODELS if (m, size) in c.P]
    common = hq.copy()
    for m in ms:
        common &= ~np.isnan(c.P[(m, size)][:, 0])
    for m in ms:
        rows.append(kl_row(m, size, hq))
        rows_c.append(kl_row(m, size, common))
kt, ktc = pd.DataFrame(rows), pd.DataFrame(rows_c)
kt.to_csv(out('kl_table.csv'), index=False)
ktc.to_csv(out('kl_table_common.csv'), index=False)

pairs = [('cnn1d_ft', 'logreg_both'), ('cnn1d_ft', 'minirocket'), ('cnn1d_ft', 'lgbm'), ('cnn1d_ft', 'tabpfn'),
         ('cnn1d_ft', 'logreg_band'), ('logreg_both', 'lgbm'), ('logreg_both', 'tabpfn'), ('tabpfn', 'minirocket'),
         ('logreg_band', 'logreg_both'), ('cnn1d', 'logreg_both')]
pe = E.cv_paired(ROOT, pairs, c=c, B=4000)
pe.to_csv(out('paired_extra.csv'), index=False)

best = []
for size in E.SIZES:
    t = ktc[(ktc.N == size) & (ktc.model != 'floor')].sort_values('kl')
    a, b = t.model.iloc[0], t.model.iloc[1]
    d = E.cv_paired(ROOT, [(a, b)], c=c, B=4000)
    d = d[d.N.astype(str) == str(size)].iloc[0]          # the row of this N (cv_paired returns every N)
    best.append(dict(N=size, best=a, kl=t.kl.iloc[0], runner_up=b, kl_runner_up=t.kl.iloc[1],
                     delta=d.delta, lo=d.ci95_lo, hi=d.ci95_hi, significant=d.significant, folds=d.folds))
pd.DataFrame(best).to_csv(out('best_by_N.csv'), index=False)

fl = E.cv_paired(ROOT, [(m, 'floor') for m in MODELS if m != 'floor'], c=c, B=4000)
first = []
for m in MODELS[1:]:
    t = fl[fl.pair == '%s - floor' % m]
    win = t[t.significant & (t.delta < 0)]
    first.append(dict(model=m, first_N_better_than_baseline=win.N.iloc[0] if len(win) else 'never',
                      worse_than_baseline_at=', '.join(str(n) for n in t[t.significant & (t.delta > 0)].N)))
pd.DataFrame(first).to_csv(out('first_beats_floor.csv'), index=False)

wins = []
for size in E.SIZES:
    ms = [m for m in MODELS if (m, size) in c.P and m != 'floor']
    for k in sorted(set(c.fold)):
        sel = hq & (c.fold == k)
        kl = {m: E.kl_per_sample(Y[sel], c.P[(m, size)][sel]).mean() for m in ms
              if not np.isnan(c.P[(m, size)][sel][:, 0]).any()}
        if kl:
            wins.append(dict(N=size, fold=k, best=min(kl, key=kl.get), n_models=len(kl)))
w = pd.DataFrame(wins)
w.groupby(['N', 'best']).size().rename('folds won').reset_index().to_csv(out('fold_wins.csv'), index=False)

m4 = ['floor', 'lgbm', 'logreg_band', 'minirocket', 'logreg_both', 'cnn1d', 'cnn1d_ft']
ok = hq.copy()
for m in m4:
    ok &= ~np.isnan(c.P[(m, 'all')][:, 0])
tabs = {m: HM.metrics_table(Y[ok], c.P[(m, 'all')][ok], pats[ok], B=1000) for m in m4}
st = HM.summary_table({m: t for m, t in tabs.items()})
st.to_csv(out('m4_common.csv'))
pd.concat({m: t for m, t in tabs.items()}).to_csv(out('m4_common_long.csv'))
with open(out('m4_common_info.json'), 'w') as fh:
    json.dump(dict(windows=int(ok.sum()), patients=len(set(pats[ok])), folds=sorted(int(f) for f in set(c.fold[ok]))), fh)

# seed spread on fold 0 (3 seeds) vs spread over folds
sd = []
for f in glob.glob(os.path.join(ROOT, '*_s[0-9].json')):
    r = json.load(open(f))
    sd.append(dict(model=r['model'] if 'model' in r else os.path.basename(f).rsplit('_n', 1)[0],
                   n=r['n_train'], seed=r['seed'], kl=r['test_kl_hq']))
sd = pd.DataFrame(sd)
sd.groupby(['model', 'n']).kl.agg(['mean', 'std', 'count']).round(4).to_csv(out('seeds_fold0.csv'))
print('done:', sorted(os.listdir(HERE)))
