"""Facts about the labels quoted in the Milestone 2 report. HMS_TRAIN_CSV = the competition's train.csv."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + '/src')
import pandas as pd, hms_data as hd

tr = pd.read_csv(os.environ.get('HMS_TRAIN_CSV', 'train.csv'))
votes = tr[hd.VOTE_COLS].astype(str).agg('-'.join, axis=1)
per_rec = votes.groupby(tr.eeg_id).nunique()
distinct = pd.DataFrame(dict(e=tr.eeg_id, v=votes)).drop_duplicates()
print('windows', len(tr))
print('recordings', tr.eeg_id.nunique())
print('patients', tr.patient_id.nunique())
print('recordings with one vote vector, percent', round(100 * (per_rec == 1).mean(), 1))
print('distinct (recording, votes) labels', len(distinct))
print('windows repeating another window of their recording, percent', round(100 * (1 - len(distinct) / len(tr)), 1))
print('windows with >= 10 votes', int((tr[hd.VOTE_COLS].sum(1) >= hd.HQ_VOTES).sum()))
