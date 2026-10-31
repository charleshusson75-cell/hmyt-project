"""MiniRocket (Dempster et al., KDD 2021) on the 16-channel signal, then a logistic regression.

MiniRocket turns each window into ~10,000 features without learning anything from the labels:
84 fixed kernels of length 9 (weights -1 and 2), each applied at several dilations to a random
subset of channels; a feature is the share of time points where the response exceeds a bias
(PPV, "proportion of positive values"). The biases are quantiles of the responses on random
TRAINING windows, so nothing is fitted on validation or test windows.

The transform is a PyTorch reimplementation of aeon's MiniRocket (aeon 1.6.0, BSD-3-Clause,
aeon/transformations/collection/convolution_based/_minirocket.py), written so it runs on the GPU
and needs no extra package on Kaggle. With the same parameters it gives the same features as
aeon (checked by tests/test_minirocket.py).

The head is a multinomial logistic regression trained on the vote shares (soft cross-entropy, the
same objective as the other models), with L2 strength 1/C chosen on the high-quality validation
windows. It runs in PyTorch (L-BFGS, full batch, GPU if any) and minimises exactly the objective of
scikit-learn's LogisticRegression on vote-weighted rows (checked by tests/test_minirocket.py); in
v1, the scikit-learn version on Kaggle's CPU took ~95% of MiniRocket's time.
"""
import time, warnings
from itertools import combinations
import numpy as np

NAME = 'minirocket'
INPUT = 'signal'

KERNELS = np.array(list(combinations(range(9), 3)), dtype=np.int64)   # positions of the three 2s (84 kernels)
CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)                                       # inverse L2 strength, weakest last


# ----------------------------------------------------------------------------- parameters (fit)
def fit_dilations(n_timepoints, n_features=10_000, max_dilations_per_kernel=32):
    """Dilations 1 .. ~(T-1)/8, log-spaced, and how many features each one gets (aeon's rule)."""
    per_kernel = n_features // 84
    true_max = min(per_kernel, max_dilations_per_kernel)
    multiplier = per_kernel / true_max
    max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
    dilations, counts = np.unique(np.logspace(0, max_exponent, true_max, base=2).astype(np.int32),
                                  return_counts=True)
    counts = (counts * multiplier).astype(np.int32)
    remainder, i = per_kernel - counts.sum(), 0
    while remainder > 0:
        counts[i] += 1
        remainder -= 1
        i = (i + 1) % len(counts)
    return dilations, counts


def quantiles(n):
    """Low-discrepancy sequence in (0, 1): the quantile used for each feature's bias."""
    return np.array([(k * ((np.sqrt(5) + 1) / 2)) % 1 for k in range(1, n + 1)], dtype=np.float32)


def response_np(x, dilation, kernel):
    """Kernel response of one window, summed over its channels. x: (channels, T) float32.
    Weight -1 at the 9 taps spaced by `dilation`, 2 at the three taps in `kernel`; zero padding."""
    n_ch, T = x.shape
    out = np.zeros((n_ch, T), np.float32)
    for tap in range(9):
        w = np.float32(2.0 if tap in kernel else -1.0)
        s = (tap - 4) * dilation                     # this tap reads x[t + s]
        if s >= 0:
            out[:, :T - s] += w * x[:, s:]
        else:
            out[:, -s:] += w * x[:, :T + s]
    return out.sum(0)


def fit_params(X, seed=0, n_features=10_000, max_dilations_per_kernel=32):
    """X: (n, channels, T) float32 training windows. Returns the parameters used by transform().
    Channel subsets follow aeon's draws exactly for a given seed (same numpy calls, same order)."""
    n, n_ch, T = X.shape
    rs = np.random.RandomState(seed)
    dilations, per_dil = fit_dilations(T, n_features, max_dilations_per_kernel)
    q = quantiles(84 * int(per_dil.sum()))
    n_comb = 84 * len(dilations)
    n_ch_comb = (2 ** rs.uniform(0, np.log2(min(n_ch, 9) + 1), n_comb)).astype(np.int32)
    channels = [rs.choice(n_ch, k, replace=False) for k in n_ch_comb]
    biases, f, comb = [], 0, 0
    for d, nf in zip(dilations, per_dil):
        for k in range(84):
            x = X[rs.randint(n)][channels[comb]]
            biases.append(np.quantile(response_np(x, int(d), KERNELS[k]), q[f:f + nf]).astype(np.float32))
            f += nf
            comb += 1
    return dict(dilations=dilations, per_dilation=per_dil, n_channels_per_comb=n_ch_comb,
                channel_indices=np.concatenate(channels).astype(np.int32),
                biases=np.concatenate(biases).astype(np.float32))


# ----------------------------------------------------------------------------- transform (PyTorch)
def transform(X, params, device=None, batch=64):
    """X: (n, channels, T) float32 -> (n, n_features) float32 in [0, 1]."""
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    n, n_ch, T = X.shape
    dil, per_dil = params['dilations'], params['per_dilation']
    W = -torch.ones(84, 1, 9)
    W[torch.arange(84)[:, None], 0, torch.as_tensor(KERNELS)] = 2.0
    W = W.to(device)
    # channel mask of each (dilation, kernel) combination and the biases, grouped by dilation
    masks, bias_d, start_c, start_f, comb = [], [], 0, 0, 0
    for nf in per_dil:
        m = np.zeros((84, n_ch), np.float32)
        for k in range(84):
            kc = params['n_channels_per_comb'][comb]
            m[k, params['channel_indices'][start_c:start_c + kc]] = 1
            start_c += kc
            comb += 1
        masks.append(torch.as_tensor(m, device=device))
        bias_d.append(torch.as_tensor(params['biases'][start_f:start_f + 84 * nf].reshape(84, nf), device=device))
        start_f += 84 * nf
    out = np.zeros((n, start_f), np.float32)
    with torch.no_grad():
        for b0 in range(0, n, batch):
            x = torch.as_tensor(np.ascontiguousarray(X[b0:b0 + batch]), dtype=torch.float32, device=device)
            B = len(x)
            feats, f0 = [], 0
            for j, (d, nf) in enumerate(zip(dil, per_dil)):
                d, pad = int(d), 4 * int(d)
                r = torch.nn.functional.conv1d(x.reshape(B * n_ch, 1, T), W, dilation=d, padding=pad)
                C = torch.einsum('bckt,kc->bkt', r.reshape(B, n_ch, 84, T), masks[j])        # (B, 84, T)
                F = torch.empty(B, 84, nf, device=device)
                for parity in (0, 1):                 # aeon: full response or valid part only, alternating
                    ks = [k for k in range(84) if (j + k) % 2 == parity]
                    Ck = C[:, ks] if parity == 0 else C[:, ks, pad:T - pad]
                    F[:, ks] = (Ck[:, :, None, :] > bias_d[j][ks][None, :, :, None]).float().mean(-1)
                feats.append(F.reshape(B, 84 * nf))
            out[b0:b0 + B] = torch.cat(feats, 1).cpu().numpy()
    return out


# ----------------------------------------------------------------------------- head
def fit_head(Ftr, Ytr, C, init=None, device=None, max_iter=500):
    """Multinomial logistic regression on soft targets: minimises mean soft cross-entropy
    + ||W||^2 / (2 C n), i.e. scikit-learn's 0.5 ||W||^2 + C * sum of vote-weighted log-losses
    divided by C n. The bias is not penalised (as in scikit-learn). Returns (W, b, iterations)."""
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.as_tensor(np.asarray(Ftr, np.float32), device=device)
    Y = torch.as_tensor(np.asarray(Ytr, np.float32), device=device)
    n, dim = X.shape
    W = (init[0].clone() if init else torch.zeros(dim, Y.shape[1], device=device)).requires_grad_(True)
    b = (init[1].clone() if init else torch.zeros(Y.shape[1], device=device)).requires_grad_(True)
    lam = 1.0 / (2.0 * C * n)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-6,
                            tolerance_change=1e-9, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = -(Y * torch.log_softmax(X @ W + b, 1)).sum(1).mean() + lam * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach(), int(opt.state[opt._params[0]].get('n_iter', 0))


def head_proba(W, b, F, batch=4096):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(F), batch):
            x = torch.as_tensor(np.asarray(F[i:i + batch], np.float32), device=W.device)
            out.append(torch.softmax(x @ W + b, 1).double().cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- model
def fit_predict(split, seed=0, n_features=10_000, Cs=CS, max_iter=500):
    from hms_eval import kl_per_sample
    t0 = time.time()
    to_nct = lambda a: np.ascontiguousarray(np.transpose(np.asarray(a, np.float32), (0, 2, 1)))
    Xtr, Xva, Xte = to_nct(split.X_train), to_nct(split.X_val), to_nct(split.X_test)
    params = fit_params(Xtr, seed, n_features)
    Ftr, Fva, Fte = (transform(a, params) for a in (Xtr, Xva, Xte))
    del Xtr, Xva, Xte                                # ~1 GB at full N, no longer needed
    t_transform = time.time() - t0
    mu, sd = Ftr.mean(0), Ftr.std(0)                 # standardise with training statistics only
    sd[sd < 1e-6] = 1.0
    Ftr, Fva, Fte = ((a - mu) / sd for a in (Ftr, Fva, Fte))

    kl = lambda Y, Q: float(kl_per_sample(Y, Q).mean())
    sel = getattr(split, 'val_hq', np.ones(len(split.Y_val), bool))
    best, val_kl, init = None, {}, None
    for C in Cs:                                     # strongest regularisation first, warm-started
        W, b, n_iter = fit_head(Ftr, split.Y_train, C, init=init, max_iter=max_iter)
        init = (W, b)
        pv = head_proba(W, b, Fva)
        val_kl[C] = round(kl(split.Y_val[sel], pv[sel]), 5)
        if best is None or val_kl[C] < best[1]:
            best = (C, val_kl[C], pv, head_proba(W, b, Fte), n_iter)
    C, _, pv, pt, n_iter = best
    import torch
    return pv, pt, dict(n_features=int(Ftr.shape[1]), dilations=[int(d) for d in params['dilations']],
                        C=C, val_kl_per_C={str(c): v for c, v in val_kl.items()}, n_iter=n_iter,
                        transform_seconds=round(t_transform, 1),
                        device='cuda' if torch.cuda.is_available() else 'cpu', torch=torch.__version__)
