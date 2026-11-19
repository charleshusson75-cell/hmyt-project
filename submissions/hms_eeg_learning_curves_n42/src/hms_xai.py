"""Milestone 4: what the models rely on.

- LightGBM (band power): SHAP values, computed exactly by LightGBM (TreeSHAP, `pred_contrib=True`).
- Logistic regression on band power: its coefficients (features are standardised, so they compare).
- CNN (signal): channel occlusion, i.e. replace one bipolar chain by its median level and measure how much
  the probability of the experts' class drops; and Grad-CAM, i.e. which moments of a window push the CNN
  towards a class (gradients at the last residual stage).
Summaries are grouped like an EEG reader would: by frequency band and by bipolar chain (left lateral,
left parasagittal, right parasagittal, right lateral).
"""
import numpy as np
from hms_data import CLASSES, CHANNELS, BANDS, FEATURE_NAMES

BAND_NAMES = [b for b, _, _ in BANDS]
CHAINS = {'left lateral': CHANNELS[0:4], 'left parasagittal': CHANNELS[4:8],
          'right parasagittal': CHANNELS[8:12], 'right lateral': CHANNELS[12:16]}


# ----------------------------------------------------------------------------- band-power models
def lgbm_shap(clf, F):
    """SHAP values (n, 6 classes, 160 features) of a fitted LightGBM classifier, in log-odds units."""
    raw = np.asarray(clf.booster_.predict(np.asarray(F), pred_contrib=True))
    nf = len(FEATURE_NAMES)
    v = raw.reshape(len(F), -1, nf + 1)[:, :, :nf]                 # last column of each class block: the bias
    out = np.zeros((len(F), len(CLASSES), nf))
    out[:, clf.classes_.astype(int)] = v
    return out


def by_band_and_chain(imp):
    """imp (6 classes, 160 features) of non-negative importances -> two tables (classes x bands, classes x
    chain groups), each row summing to 1: the share of the class's importance in each band or region."""
    import pandas as pd
    names = np.array(FEATURE_NAMES)
    band = np.array([n.split('_')[1] for n in names])
    chan = np.array([n.split('_', 2)[2] for n in names])
    tb = pd.DataFrame({b: imp[:, band == b].sum(1) for b in BAND_NAMES}, index=CLASSES)
    tc = pd.DataFrame({g: imp[:, np.isin(chan, ch)].sum(1) for g, ch in CHAINS.items()}, index=CLASSES)
    share = lambda t: t.div(t.sum(1).replace(0, np.nan), axis=0).round(3)   # a class with no importance: NaN
    return share(tb), share(tc)


def top_features(weights, k=5):
    """For each class, the k features with the largest positive weight (signed: what raises the class)."""
    import pandas as pd
    names = np.array(FEATURE_NAMES)
    return pd.DataFrame({c: ['%s (%+.2f)' % (names[i], weights[j, i]) for i in np.argsort(-weights[j])[:k]]
                         for j, c in enumerate(CLASSES)})


# ----------------------------------------------------------------------------- CNN
def _to_nct(X):
    return np.ascontiguousarray(np.transpose(np.asarray(X, np.float32), (0, 2, 1)))


def channel_occlusion(net, X, Y, device, batch=256):
    """Drop of the probability of the experts' majority class when one bipolar chain is replaced by its
    median level (0 after the per-channel normalisation), averaged over the windows of each majority class.
    Returns a DataFrame classes x chains (larger = the CNN relies more on that chain for that class)."""
    import pandas as pd
    from models.deep import _predict
    Xn = _to_nct(X)
    true = np.asarray(Y).argmax(1)
    base = _predict(net, Xn, device, batch)[np.arange(len(Xn)), true]
    drop = np.zeros((len(CLASSES), len(CHANNELS)))
    for c in range(len(CHANNELS)):
        Xo = Xn.copy()
        Xo[:, c, :] = 0.0
        d = base - _predict(net, Xo, device, batch)[np.arange(len(Xn)), true]
        for j in range(len(CLASSES)):
            if (true == j).any():
                drop[j, c] = d[true == j].mean()
    return pd.DataFrame(drop, index=CLASSES, columns=CHANNELS).round(4)


def gradcam(net, x, cls, device):
    """Grad-CAM over time for one window x (T, channels) and class index cls: weights = time-averaged gradients
    of the class logit at the last residual stage. Returns an array of length T in [0, 1]."""
    import torch
    import torch.nn as nn
    mods = list(net.children())
    last = max(i for i, m in enumerate(mods) if isinstance(m, nn.Module) and type(m).__name__ == 'Block')
    store = {}

    def keep(m, i, o):                                # the activation, and its gradient when it comes back
        store['a'] = o
        o.register_hook(lambda g: store.__setitem__('g', g))
    h = mods[last].register_forward_hook(keep)
    net.eval()
    xt = torch.as_tensor(_to_nct(x[None]), device=device).requires_grad_(True)
    try:
        net.zero_grad()
        net(xt)[0, cls].backward()
    finally:
        h.remove()
    a, g = store['a'][0].detach().float(), store['g'][0].detach().float()
    cam = torch.relu((g.mean(1, keepdim=True) * a).sum(0)).cpu().numpy()
    T = x.shape[0]
    cam = np.interp(np.linspace(0, len(cam) - 1, T), np.arange(len(cam)), cam)
    return cam / cam.max() if cam.max() > 0 else cam


def examples(Y, P, k=1):
    """For each class, the index of the k test windows where the experts agree most on that class and the
    model predicts it with the highest probability: clear cases, to show what the model looks at."""
    Y, P = np.asarray(Y), np.asarray(P)
    out = {}
    for j, c in enumerate(CLASSES):
        ok = np.flatnonzero((Y.argmax(1) == j) & (P.argmax(1) == j))
        if len(ok):
            out[c] = ok[np.argsort(-(Y[ok, j] + P[ok, j]))[:k]].tolist()
    return out


def error_examples(Y, P, k=6):
    """Up to k misclassified windows (experts' majority class != model's most likely class), clear cases
    of each confusion: within a (experts' class, model's class) pair, the window where the experts agree
    most and the model is most confident. First a missed seizure and a false seizure (the two costly
    errors), then the most frequent confusions. Returns a list of indices into Y."""
    Y, P = np.asarray(Y), np.asarray(P)
    t, p = Y.argmax(1), P.argmax(1)
    wrong = t != p
    s = CLASSES.index('seizure')

    def pick(mask):
        idx = np.flatnonzero(mask)
        return int(idx[np.argmax(Y[idx, t[idx]] + P[idx, p[idx]])]) if len(idx) else None
    out = [i for i in (pick(wrong & (t == s)), pick(wrong & (p == s))) if i is not None]
    pairs = [(a, b) for a in range(len(CLASSES)) for b in range(len(CLASSES)) if a != b and s not in (a, b)]
    for a, b in sorted(pairs, key=lambda ab: -int((wrong & (t == ab[0]) & (p == ab[1])).sum())):
        if len(out) >= k:
            break
        i = pick(wrong & (t == a) & (p == b))
        if i is not None:
            out.append(i)
    return out


# ----------------------------------------------------------------------------- plots
def plot_heatmap(t, title, ax=None, fmt='%.2f', cmap='Purples'):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(0.62 * t.shape[1] + 2.2, 0.42 * t.shape[0] + 1.4))[1]
    v = t.values.astype(float)
    ax.imshow(v, cmap=cmap, aspect='auto', vmin=0, vmax=max(v.max(), 1e-9))
    ax.set_xticks(range(t.shape[1]))
    ax.set_xticklabels(t.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(t.shape[0]))
    ax.set_yticklabels(t.index, fontsize=8)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, fmt % v[i, j], ha='center', va='center', fontsize=7,
                    color='white' if v[i, j] > 0.6 * v.max() else '#1f1f1e')
    ax.set_title(title, fontsize=9, loc='left')
    return ax


def plot_gradcam(x, cam, title, ax=None, fs=200):
    """The 16 chains of one window (top to bottom), the background shaded by Grad-CAM (darker = more weight)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(9, 5.2))[1]
    T = x.shape[0]
    t = np.arange(T) / fs
    ax.imshow(cam[None, :], aspect='auto', cmap='Oranges', alpha=0.55, vmin=0, vmax=1,
              extent=(0, T / fs, -0.5, len(CHANNELS) - 0.5))
    span = np.percentile(np.abs(x), 99) or 1.0
    for c in range(len(CHANNELS)):
        ax.plot(t, len(CHANNELS) - 1 - c + 0.45 * x[:, c] / span, color='#1f1f1e', lw=0.5)
    ax.set_yticks(range(len(CHANNELS)))
    ax.set_yticklabels(CHANNELS[::-1], fontsize=7)
    ax.set_xlabel('seconds')
    ax.set_title(title, fontsize=9, loc='left')
    return ax
