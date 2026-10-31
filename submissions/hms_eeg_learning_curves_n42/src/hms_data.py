"""HMS data, protocol v2: preparation (run once), loading, and one training split per (N, seed).

    prepare(comp_dir, out_dir)        parquet -> X.npy, F.npy, meta.csv, prep_config.json
    data  = load(out_dir)             data.X, data.F, data.meta
    split = make_split(data, N, seed) what every model receives

Windows. Every recording has one PRIMARY window, drawn at random (fixed seed): N counts these, one
per labelled recording. Its other labelled windows whose central 10 s do not overlap a kept one are
stored too (`is_primary` False): they carry the same annotation in ~89% of recordings, so they are
extra signal, not extra labels. They serve the high-quality test set now and a "+crops" training
option later.

Split, decided ONCE in prepare() (column `split` of meta.csv): patient-disjoint AND class-stratified
(StratifiedGroupKFold on the primary windows' majority class), as in Jing et al. 2023 and Sun et al.
2025. All windows of a patient follow the patient. make_split() only chooses which N training
recordings are used.

Test (and validation) windows: the primary window of every recording ("all" set, one per recording)
plus every stored window with at least 10 votes ("hq" set: the reliable labels, the primary metric).
"""
import os, json, time, warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd

PREP_VERSION = 2
CLASSES = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
VOTE_COLS = ['%s_vote' % c for c in CLASSES]
ELECTRODES = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
              'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
BIPOLAR = [('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),    # LL left lateral
           ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),    # LP left parasagittal
           ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),    # RP right parasagittal
           ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2')]    # RL right lateral
CHANNELS = ['%s-%s' % p for p in BIPOLAR]
BANDS = [('delta', 0.5, 4.0), ('theta', 4.0, 8.0), ('alpha', 8.0, 13.0),
         ('beta', 13.0, 30.0), ('gamma', 30.0, 45.0)]
FEATURE_NAMES = (['log_%s_%s' % (b, ch) for ch in CHANNELS for b, _, _ in BANDS] +
                 ['rel_%s_%s' % (b, ch) for ch in CHANNELS for b, _, _ in BANDS])

FS_RAW = 200          # HMS acquisition rate
FS_MODEL = 200        # kept at the native rate (v1 used 100 Hz, an ECG-model input length)
WIN_S = 50            # each annotated sample is a 50 s window ...
LABEL_S = 10          # ... of which the experts labelled the central 10 s
CLIP_UV = 1024.0
NOTCH_HZ = 60.0       # US mains
BANDPASS = (0.3, 75.0)  # CBraMod's pretraining filter (TUH): one signal for every model
HQ_VOTES = 10         # windows with >= 10 votes = high-quality labels (Jing et al. 2023)
HIGH_AGREEMENT = HQ_VOTES   # older name


# ----------------------------------------------------------------- choice of windows
def select_windows(train, seed=0, min_gap=LABEL_S):
    """Primary window per recording (random, fixed seed) + the recording's other windows whose
    label centre is at least `min_gap` s from every window already kept (primary first)."""
    k = np.random.default_rng(seed).random(len(train))
    t = train.assign(_k=k)
    prim_idx = t.sort_values(['eeg_id', '_k']).groupby('eeg_id', sort=True).head(1).index
    t['is_primary'] = False
    t.loc[prim_idx, 'is_primary'] = True
    t = t.sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    keep = np.zeros(len(t), bool)
    eid, off, prim = (t.eeg_id.to_numpy(), t.eeg_label_offset_seconds.to_numpy(), t.is_primary.to_numpy())
    starts = np.flatnonzero(np.r_[True, eid[1:] != eid[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(t)]):
        kept = [off[a:b][prim[a:b]][0]]
        keep[a + np.flatnonzero(prim[a:b])[0]] = True
        for j in range(a, b):
            if not prim[j] and all(abs(off[j] - o) >= min_gap for o in kept):
                kept.append(off[j])
                keep[j] = True
    return t[keep].drop(columns='_k').reset_index(drop=True)


# ----------------------------------------------------------------- signal processing
def to_bipolar(raw):
    """16 double-banana chains. A missing electrode silences exactly its two chains."""
    idx = {e: i for i, e in enumerate(ELECTRODES)}
    bip = np.stack([raw[:, idx[a]] - raw[:, idx[b]] for a, b in BIPOLAR], axis=1)
    missing = float(np.isnan(bip).mean())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN chain
        med = np.nanmedian(bip, axis=0)
    bip = np.nan_to_num(bip - np.nan_to_num(med), nan=0.0)
    return np.clip(bip, -CLIP_UV, CLIP_UV), missing


def filter_signal(x, fs=FS_RAW):
    """60 Hz notch + 0.3-75 Hz Butterworth(3), both zero-phase."""
    from scipy.signal import iirnotch, filtfilt, butter, sosfiltfilt
    b, a = iirnotch(NOTCH_HZ, 30.0, fs)
    x = filtfilt(b, a, x, axis=0)
    return sosfiltfilt(butter(3, BANDPASS, btype='bandpass', fs=fs, output='sos'), x, axis=0)


def centre(x, fs, seconds=LABEL_S):
    """The labelled central window. Filtering runs on the full 50 s first, so edge
    effects fall outside what is kept."""
    n = int(seconds * fs)
    a = (len(x) - n) // 2
    return x[a:a + n]


def band_features(x, fs=FS_RAW):
    """160 features: log10 absolute and relative power, 5 bands x 16 chains, on the
    filtered but NOT normalised signal. Welch, 2 s segments."""
    from scipy.signal import welch
    f, pxx = welch(x, fs=fs, nperseg=2 * fs, axis=0)
    df_ = f[1] - f[0]
    bp = np.stack([pxx[(f >= lo) & (f < hi)].sum(0) * df_ for _, lo, hi in BANDS])
    rel = bp / (bp.sum(0, keepdims=True) + 1e-12)
    return np.concatenate([np.log10(bp + 1e-6).T.ravel(), rel.T.ravel()]).astype(np.float32)


def _process_recording(comp_dir, eeg_id, offsets):
    """Read one recording once; return [(x, f, missing, error)] for each window offset."""
    out = []
    try:
        df = pd.read_parquet(os.path.join(comp_dir, 'train_eegs', '%d.parquet' % eeg_id))
        missing = [e for e in ELECTRODES if e not in df.columns]
        if missing:
            raise KeyError('lacks electrodes %s' % missing)
        arr = df[ELECTRODES].to_numpy(np.float64)
    except Exception as e:
        return [(None, None, None, '%s: %r' % (eeg_id, e))] * len(offsets)
    for off in offsets:
        try:
            a, n = int(round(off * FS_RAW)), WIN_S * FS_RAW
            w = arr[a:a + n]
            if len(w) != n:
                raise ValueError('window at %.1f s has %d samples, expected %d' % (off, len(w), n))
            bip, miss = to_bipolar(w)
            x = centre(filter_signal(bip), FS_MODEL)
            out.append((x.astype(np.float16), band_features(x, FS_MODEL), miss, None))
        except Exception as e:
            out.append((None, None, None, '%s@%.0f: %r' % (eeg_id, off, e)))
    return out


# ----------------------------------------------------------------- split
def majority_class(meta):
    return meta[VOTE_COLS].to_numpy().argmax(1)


def assign_split(meta, n_folds=10, seed=0, test_fold=0, val_fold=1):
    """Patient-disjoint, class-stratified folds (StratifiedGroupKFold on the primary windows):
    fold 0 = test, fold 1 = val, folds 2-9 = train pool. Every window follows its patient."""
    from sklearn.model_selection import StratifiedGroupKFold
    p = meta[meta.is_primary]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = {}
    for k, (_, idx) in enumerate(sgkf.split(p, majority_class(p), groups=p.patient_id)):
        for pat in p.patient_id.iloc[idx]:
            fold_of[pat] = k
    out = meta.copy()
    out['fold'] = out['patient_id'].map(fold_of).astype(int)
    out['split'] = np.select([out.fold == test_fold, out.fold == val_fold], ['test', 'val'], 'train')
    check_disjoint(out)
    return out


def check_disjoint(meta):
    """No patient and no recording in two splits: train, val and test are independent."""
    for key in ('patient_id', 'eeg_id'):
        tr, va, te = (set(meta.loc[meta['split'] == s, key]) for s in ('train', 'val', 'test'))
        assert tr.isdisjoint(te) and tr.isdisjoint(va) and va.isdisjoint(te), 'a value of %s is in two splits' % key


def describe_split(meta):
    """Per split: recordings, patients, high-quality windows, and the class mix of the primary windows."""
    out = {}
    for k, v in meta.groupby('split'):
        p = v[v.is_primary]
        mix = np.bincount(majority_class(p), minlength=len(CLASSES)) / max(len(p), 1)
        out[k] = dict(recordings=int(len(p)), patients=int(v.patient_id.nunique()),
                      windows_stored=int(len(v)), hq_windows=int((v.total_votes >= HQ_VOTES).sum()),
                      **{c: round(float(m), 3) for c, m in zip(CLASSES, mix)})
    return out


# ----------------------------------------------------------------- preparation
def prepare(comp_dir, out_dir, seed=0, limit=None, n_jobs=-1, chunk=500, max_fail=0.01):
    """limit: number of recordings (for a quick test)."""
    from joblib import Parallel, delayed
    rows = select_windows(pd.read_csv(os.path.join(comp_dir, 'train.csv')), seed)
    if limit:
        rows = rows[rows.eeg_id.isin(rows.eeg_id.unique()[:int(limit)])].reset_index(drop=True)
    n = len(rows)
    os.makedirs(out_dir, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(out_dir, 'X.npy'), mode='w+', dtype=np.float16,
                                  shape=(n, LABEL_S * FS_MODEL, len(BIPOLAR)))
    F = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    miss, ok, fails, t0 = np.zeros(n, np.float32), np.zeros(n, bool), [], time.time()
    groups = [(int(e), g.index.to_numpy(), g.eeg_label_offset_seconds.to_numpy())
              for e, g in rows.groupby('eeg_id', sort=True)]
    for s in range(0, len(groups), chunk):
        part = groups[s:s + chunk]
        res = Parallel(n_jobs=n_jobs)(delayed(_process_recording)(comp_dir, e, offs) for e, _, offs in part)
        for (_, idx, _), windows in zip(part, res):
            for i, (x, f, m, err) in zip(idx, windows):
                if err:
                    fails.append(err)
                    continue
                X[i], F[i], miss[i], ok[i] = x, f, m, True
        print('  %6d / %d recordings | %d windows failed | %.0f s'
              % (min(s + chunk, len(groups)), len(groups), len(fails), time.time() - t0), flush=True)
    X.flush()
    if len(fails) > max_fail * n:
        raise RuntimeError('%d of %d windows failed; first: %s' % (len(fails), n, fails[:3]))
    meta = rows[['eeg_id', 'eeg_sub_id', 'eeg_label_offset_seconds', 'patient_id', 'expert_consensus',
                 'is_primary'] + VOTE_COLS].copy()
    meta['total_votes'] = meta[VOTE_COLS].sum(1).astype(int)
    meta['missing_frac'] = miss
    meta['xrow'] = np.arange(n)                      # row in X.npy / F.npy
    meta = meta[ok].reset_index(drop=True)
    lost = set(rows.eeg_id) - set(meta.eeg_id[meta.is_primary])
    if lost:                                         # a recording whose primary window failed: drop it
        meta = meta[~meta.eeg_id.isin(lost)].reset_index(drop=True)
    if (meta.total_votes <= 0).any():
        raise ValueError('rows with zero votes: %d' % int((meta.total_votes <= 0).sum()))
    meta = assign_split(meta, seed=seed)
    np.save(os.path.join(out_dir, 'F.npy'), F)
    meta.to_csv(os.path.join(out_dir, 'meta.csv'), index=False)
    desc = describe_split(meta)
    with open(os.path.join(out_dir, 'prep_config.json'), 'w') as fh:
        json.dump(dict(version=PREP_VERSION, seed=seed, fs_raw=FS_RAW, fs_model=FS_MODEL, win_s=WIN_S,
                       label_s=LABEL_S, channels=CHANNELS, clip_uv=CLIP_UV, notch_hz=NOTCH_HZ,
                       bandpass=BANDPASS, bands=BANDS, feature_names=FEATURE_NAMES, hq_votes=HQ_VOTES,
                       windows=int(len(meta)), recordings=int(meta.is_primary.sum()),
                       n_failed=len(fails), recordings_dropped=len(lost), failures=fails[:200], split=desc),
                  fh, indent=1)
    print('done: %d recordings, %d windows stored, %d windows failed | %.0f s'
          % (meta.is_primary.sum(), len(meta), len(fails), time.time() - t0))
    return meta


# ----------------------------------------------------------------- loading and splits
def load(d):
    with open(os.path.join(d, 'prep_config.json')) as fh:
        cfg = json.load(fh)
    if cfg.get('version') != PREP_VERSION:
        raise ValueError('%s holds prepared data of version %s; this code needs version %d: run prepare() again'
                         % (d, cfg.get('version', 1), PREP_VERSION))
    X = np.load(os.path.join(d, 'X.npy'), mmap_mode='r')
    F = np.load(os.path.join(d, 'F.npy'))
    meta = pd.read_csv(os.path.join(d, 'meta.csv'))
    if meta.xrow.max() >= len(X) or len(F) != len(X):
        raise ValueError('X / F / meta do not match')
    check_disjoint(meta)
    return SimpleNamespace(X=X, F=F, meta=meta, config=cfg)


def soft_targets(meta):
    v = meta[VOTE_COLS].to_numpy(np.float64)
    return (v / v.sum(1, keepdims=True)).astype(np.float32)


def subsample(pool, n, seed):
    """n rows of the train pool; n=None means all. Nested: for one seed, the 100 are
    inside the 1000."""
    pool = np.asarray(pool)
    if n is None or n >= len(pool):
        return np.sort(pool)
    return np.sort(np.random.default_rng(seed).permutation(pool)[:int(n)])


def make_split(data, n_train, seed, signals=True, clip=20.0, stat_windows=2000):
    """Everything a model receives. Y_* are the soft targets (vote proportions).
    Train: the primary windows of N training recordings. Val and test: primary windows plus
    high-quality windows; masks *_primary and *_hq say which. X_* (n, time, channels) are
    standardised per channel (median/IQR) using the TRAINING windows only."""
    meta = data.meta
    split, prim = meta['split'].to_numpy(), meta['is_primary'].to_numpy()
    hq = meta['total_votes'].to_numpy() >= HQ_VOTES
    tr = subsample(np.flatnonzero(prim & (split == 'train')), n_train, seed)
    va = np.flatnonzero((split == 'val') & (prim | hq))
    te = np.flatnonzero((split == 'test') & (prim | hq))
    Y, xrow = soft_targets(meta), meta['xrow'].to_numpy()
    s = SimpleNamespace(
        n_train=len(tr), seed=seed, rows_train=tr, rows_val=va, rows_test=te,
        n_patients_train=int(meta['patient_id'].iloc[tr].nunique()),
        Y_train=Y[tr], Y_val=Y[va], Y_test=Y[te],
        F_train=data.F[xrow[tr]], F_val=data.F[xrow[va]], F_test=data.F[xrow[te]],
        val_primary=prim[va], val_hq=hq[va], test_primary=prim[te], test_hq=hq[te],
        ids_test=meta['eeg_id'].to_numpy()[te], subids_test=meta['eeg_sub_id'].to_numpy()[te],
        patients_test=meta['patient_id'].to_numpy()[te], patients_val=meta['patient_id'].to_numpy()[va],
        total_votes_test=meta['total_votes'].to_numpy()[te],
        X_train=None, X_val=None, X_test=None)
    if signals:
        xtr = np.asarray(data.X[xrow[tr]], np.float32)
        pick = np.random.default_rng(seed).choice(len(tr), min(len(tr), stat_windows), replace=False)
        med = np.median(xtr[pick], axis=(0, 1))
        q75, q25 = np.percentile(xtr[pick], [75, 25], axis=(0, 1))
        iqr = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
        norm = lambda a: np.clip((np.asarray(a, np.float32) - med) / iqr, -clip, clip).astype(np.float32)
        s.X_train, s.X_val, s.X_test = norm(xtr), norm(data.X[xrow[va]]), norm(data.X[xrow[te]])
        s.norm_stats = dict(median=med, iqr=iqr)
    return s
