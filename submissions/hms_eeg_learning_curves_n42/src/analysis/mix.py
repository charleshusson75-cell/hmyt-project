import sys, json
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the team folder
TRAIN_CSV = os.environ.get('HMS_TRAIN_CSV', 'train.csv')   # train.csv of the competition
sys.path.insert(0, ROOT + '/src')
import numpy as np, pandas as pd, hms_data as hd
meta = hd.select_windows(pd.read_csv(TRAIN_CSV), seed=0)
meta['total_votes'] = meta[hd.VOTE_COLS].sum(1).astype(int)
meta = hd.assign_split(meta)
Y = hd.soft_targets(meta); hq = (meta.total_votes >= 10).to_numpy(); sp = meta.split.to_numpy(); pr = meta.is_primary.to_numpy()
out = {'training windows (all votes)': Y[(sp == 'train') & pr].mean(0), 'training windows, >=10 votes': Y[(sp == 'train') & hq].mean(0),
       'test windows, >=10 votes': Y[(sp == 'test') & hq].mean(0), 'test windows, <10 votes (primary)': Y[(sp == 'test') & pr & ~hq].mean(0)}
out = {k: [round(float(x), 3) for x in v] for k, v in out.items()}
print(hd.CLASSES); [print(k, v) for k, v in out.items()]
json.dump(dict(classes=list(hd.CLASSES), mix=out), open(os.path.dirname(os.path.abspath(__file__)) + '/mix.json', 'w'), indent=1)
