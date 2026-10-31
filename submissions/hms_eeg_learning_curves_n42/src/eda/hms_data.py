"""HMS data: preparation (run once), loading, and one training split per (N, seed).

    prepare(comp_dir, out_dir)        parquet -> X.npy, F.npy, meta.csv, prep_config.json
    data  = load(out_dir)             data.X, data.F, data.meta
    split = make_split(data, N, seed) what every model receives

The split is decided ONCE in prepare() (column `split` of meta.csv): patient-disjoint, and
val/test never change. make_split() only chooses which N training windows are used.
"""
import os, json, time, warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd

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
WIN_S = 50            # each annotated sample is a 50 s window ...
LABEL_S = 10          # ... of which the experts labelled the central 10 s
FS_MODEL = 100        # PTB-XL rate: 10 s = 1000 samples, the ECG model's input length
CLIP_UV = 1024.0
NOTCH_HZ = 60.0       # US mains
BANDPASS = (0.5, 45.0)
HIGH_AGREEMENT = 10   # total_votes >= 10 -> 'high-agreement' test window


# ----------------------------------------------------------------- preparation
def select_rows(train, seed=0):
    """One labelled window per recording, drawn at random (fixed seed). Overlapping windows
    of one recording are near-duplicates; random rather than max-votes keeps the natural
    mix of low- and high-agreement labels."""
    k = np.random.default_rng(seed).random(len(train))
    t = train.assign(_k=k).sort_values(['eeg_id', '_k'])
    return t.groupby('eeg_id', sort=True).head(1).drop(columns='_k').reset_index(drop=True)


def read_window(path, offset_s):
    """The annotated 50 s window, (10000, 19), referential, NaNs kept."""
    df = pd.read_parquet(path)
    missing = [e for e in ELECTRODES if e not in df.columns]
    if missing:
        raise KeyError('%s lacks electrodes %s (columns: %s)' % (path, missing, list(df.columns)))
    a, n = int(round(offset_s * FS_RAW)), WIN_S * FS_RAW
    w = df[ELECTRODES].to_numpy(np.float64)[a:a + n]
    if len(w) != n:
        raise ValueError('window at %.1f s has %d samples, expected %d' % (offset_s, len(w), n))
    return w


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
    """60 Hz notch + 0.5-45 Hz Butterworth(3), both zero-phase."""
    from scipy.signal import iirnotch, filtfilt, butter, sosfiltfilt
    b, a = iirnotch(NOTCH_HZ, 30.0, fs)
    x = filtfilt(b, a, x, axis=0)
    return sosfiltfilt(butter(3, BANDPASS, btype='bandpass', fs=fs, output='sos'), x, axis=0)


def to_model_rate(x, fs=FS_RAW):
    from scipy.signal import decimate
    q = fs // FS_MODEL
    return x if q == 1 else decimate(x, q, axis=0, zero_phase=True)


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


def _process(comp_dir, eeg_id, offset_s):
    try:
        raw = read_window(os.path.join(comp_dir, 'train_eegs', '%d.parquet' % eeg_id), offset_s)
        bip, miss = to_bipolar(raw)
        filt = filter_signal(bip)
        x = centre(to_model_rate(filt), FS_MODEL).astype(np.float16)
        return x, band_features(centre(filt, FS_RAW)), miss, None
    except Exception as e:
        return None, None, None, '%s: %r' % (eeg_id, e)


def assign_split(meta, n_folds=10, seed=0, test_fold=0, val_fold=1):
    """Patient-disjoint folds balanced by window count: fold 0 = test, fold 1 = val,
    folds 2-9 = train pool."""
    rng = np.random.default_rng(seed)
    counts = meta.groupby('patient_id').size()
    pats, cnt = counts.index.to_numpy(), counts.to_numpy()
    load_, fold_of = np.zeros(n_folds, int), {}
    for k in np.lexsort((rng.random(len(pats)), -cnt)):
        f = int(np.argmin(load_))
        fold_of[pats[k]] = f
        load_[f] += cnt[k]
    out = meta.copy()
    out['fold'] = out['patient_id'].map(fold_of).astype(int)
    out['split'] = np.select([out.fold == test_fold, out.fold == val_fold], ['test', 'val'], 'train')
    check_disjoint(out)
    return out


def check_disjoint(meta):
    s = meta.groupby('patient_id')['split'].nunique()
    if (s > 1).any():
        raise AssertionError('%d patients appear in more than one split' % int((s > 1).sum()))


def describe_split(meta):
    return {k: dict(windows=int(len(v)), patients=int(v.patient_id.nunique()),
                    high_agreement=int((v.total_votes >= HIGH_AGREEMENT).sum()))
            for k, v in meta.groupby('split')}


def prepare(comp_dir, out_dir, seed=0, limit=None, n_jobs=-1, chunk=1000, max_fail=0.01):
    from joblib import Parallel, delayed
    rows = select_rows(pd.read_csv(os.path.join(comp_dir, 'train.csv')), seed)
    if limit:
        rows = rows.iloc[:int(limit)].reset_index(drop=True)
    n = len(rows)
    X = np.zeros((n, LABEL_S * FS_MODEL, len(BIPOLAR)), np.float16)
    F = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    miss, ok, fails, t0 = np.zeros(n, np.float32), np.zeros(n, bool), [], time.time()
    for s in range(0, n, chunk):
        part = rows.iloc[s:s + chunk]
        res = Parallel(n_jobs=n_jobs)(
            delayed(_process)(comp_dir, int(r.eeg_id), float(r.eeg_label_offset_seconds))
            for r in part.itertuples(index=False))
        for j, (x, f, m, err) in enumerate(res):
            if err:
                fails.append(err)
                continue
            X[s + j], F[s + j], miss[s + j], ok[s + j] = x, f, m, True
        print('  %6d / %d windows | %d failed | %.0f s'
              % (min(s + chunk, n), n, len(fails), time.time() - t0), flush=True)
    if len(fails) > max_fail * n:
        raise RuntimeError('%d of %d windows failed; first: %s' % (len(fails), n, fails[:3]))
    rows, X, F, miss = rows.loc[ok].reset_index(drop=True), X[ok], F[ok], miss[ok]
    meta = rows[['eeg_id', 'eeg_sub_id', 'eeg_label_offset_seconds', 'patient_id',
                 'expert_consensus'] + VOTE_COLS].copy()
    meta['total_votes'] = meta[VOTE_COLS].sum(1).astype(int)
    if (meta.total_votes <= 0).any():
        raise ValueError('rows with zero votes: %d' % int((meta.total_votes <= 0).sum()))
    meta['missing_frac'] = miss
    meta = assign_split(meta, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'F.npy'), F)
    meta.to_csv(os.path.join(out_dir, 'meta.csv'), index=False)
    with open(os.path.join(out_dir, 'prep_config.json'), 'w') as fh:
        json.dump(dict(seed=seed, fs_raw=FS_RAW, fs_model=FS_MODEL, win_s=WIN_S, label_s=LABEL_S,
                       channels=CHANNELS, clip_uv=CLIP_UV, notch_hz=NOTCH_HZ, bandpass=BANDPASS,
                       bands=BANDS, feature_names=FEATURE_NAMES, n=int(len(meta)),
                       n_failed=len(fails), failures=fails[:200], split=describe_split(meta)),
                  fh, indent=1)
    print('done: %d windows, %d failed | %s' % (len(meta), len(fails), describe_split(meta)))
    return meta


# ----------------------------------------------------------------- loading and splits
def load(d):
    X = np.load(os.path.join(d, 'X.npy'), mmap_mode='r')
    F = np.load(os.path.join(d, 'F.npy'))
    meta = pd.read_csv(os.path.join(d, 'meta.csv'))
    if not len(X) == len(F) == len(meta):
        raise ValueError('X / F / meta lengths differ')
    check_disjoint(meta)
    return SimpleNamespace(X=X, F=F, meta=meta)


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
    X_* are standardised per channel (median/IQR) using the TRAINING windows only."""
    meta = data.meta
    split = meta['split'].to_numpy()
    tr = subsample(np.flatnonzero(split == 'train'), n_train, seed)
    va, te = np.flatnonzero(split == 'val'), np.flatnonzero(split == 'test')
    Y = soft_targets(meta)
    s = SimpleNamespace(
        n_train=len(tr), seed=seed, rows_train=tr, rows_val=va, rows_test=te,
        n_patients_train=int(meta['patient_id'].iloc[tr].nunique()),
        Y_train=Y[tr], Y_val=Y[va], Y_test=Y[te],
        F_train=data.F[tr], F_val=data.F[va], F_test=data.F[te],
        ids_test=meta['eeg_id'].to_numpy()[te],
        total_votes_test=meta['total_votes'].to_numpy()[te],
        X_train=None, X_val=None, X_test=None)
    if signals:
        xtr = np.asarray(data.X[tr], np.float32)
        pick = np.random.default_rng(seed).choice(len(tr), min(len(tr), stat_windows), replace=False)
        med = np.median(xtr[pick], axis=(0, 1))
        q75, q25 = np.percentile(xtr[pick], [75, 25], axis=(0, 1))
        iqr = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
        norm = lambda a: np.clip((np.asarray(a, np.float32) - med) / iqr, -clip, clip).astype(np.float32)
        s.X_train, s.X_val, s.X_test = norm(xtr), norm(data.X[va]), norm(data.X[te])
    return s
