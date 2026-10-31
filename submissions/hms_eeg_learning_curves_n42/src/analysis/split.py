import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the team folder
TRAIN_CSV = os.environ.get('HMS_TRAIN_CSV', 'train.csv')   # train.csv of the competition
sys.path.insert(0, ROOT + '/src')
import numpy as np, pandas as pd, hms_data as hd
pd.set_option('display.width', 200)
meta = hd.select_windows(pd.read_csv(TRAIN_CSV), seed=0)
meta['total_votes'] = meta[hd.VOTE_COLS].sum(1).astype(int)
meta = hd.assign_split(meta)
print(pd.DataFrame(hd.describe_split(meta)).T.to_string())
p = meta[meta.is_primary]
full = np.bincount(hd.majority_class(p), minlength=6) / len(p)
print('all recordings', dict(zip(hd.CLASSES, full.round(3))))
r = p.groupby('patient_id').size().sort_values(ascending=False)
print('recordings per patient: top', r.head(8).tolist(), 'median', r.median())
for s in ['test','val']:
    v = p[p.split == s]; c = v.groupby('patient_id').size().sort_values(ascending=False)
    top = c.index[0]; print(s, 'biggest patient', top, c.iloc[0], 'recordings; its majority mix',
        dict(zip(hd.CLASSES, (np.bincount(hd.majority_class(v[v.patient_id == top]), minlength=6)).tolist())))
hq = meta[(meta.split == 'test') & (meta.total_votes >= 10)]
print('test hq: windows', len(hq), 'recordings', hq.eeg_id.nunique(), 'patients', hq.patient_id.nunique())
x = hq[hq.patient_id == 56885]
print('patient 56885: hq windows', len(x), 'recordings', x.eeg_id.nunique(), 'distinct vote vectors', x[hd.VOTE_COLS].drop_duplicates().shape[0],
      'majority', pd.Series(hd.majority_class(x)).map(dict(enumerate(hd.CLASSES))).value_counts().to_dict())
