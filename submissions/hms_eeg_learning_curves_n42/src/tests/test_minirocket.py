"""Checks for models/minirocket.py against the reference implementation (aeon's MiniRocket).

    python tests/test_minirocket.py          (from the code/ folder; needs torch and scikit-learn, and aeon for checks 3-5)

1. expand_votes gives the same rows as the previous implementation.
2. The PyTorch logistic-regression head gives scikit-learn's probabilities (same objective).
3. With aeon's fitted parameters, our PyTorch transform gives aeon's features.
4. For a given seed, our channel subsets are aeon's.
5. Each of aeon's biases is the quantile of our kernel response on one of the training windows.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import minirocket as M
from models.lgbm import expand_votes

rng = np.random.default_rng(0)

# 1 ------------------------------------------------------------------------------------------
F = rng.normal(size=(50, 7)).astype(np.float32)
Y = rng.dirichlet(np.ones(6) * 0.3, 50)
Y[Y < 0.1] = 0
Y /= Y.sum(1, keepdims=True)
old_keep = np.asarray(Y, np.float64).ravel() > 0
old = (np.repeat(F, 6, axis=0)[old_keep], np.tile(np.arange(6), 50)[old_keep], Y.ravel()[old_keep])
new = expand_votes(F, Y)
assert all(np.array_equal(a, b) for a, b in zip(old, new)), 'expand_votes changed its output'
print('1. expand_votes: same rows, classes and weights as before (%d rows)' % len(new[0]))

# 2 ------------------------------------------------------------------------------------------
# the PyTorch head minimises the same objective as scikit-learn's LogisticRegression on vote-weighted rows
from sklearn.linear_model import LogisticRegression
Fh = rng.normal(size=(300, 40)).astype(np.float32)
Wtrue = rng.normal(size=(40, 6))
Pt = np.exp(Fh @ Wtrue * 0.3); Pt /= Pt.sum(1, keepdims=True)
Yh = np.stack([rng.multinomial(5, p) for p in Pt]).astype(np.float64)
Yh /= Yh.sum(1, keepdims=True)
for C in (0.01, 1.0):
    Xe, ye, we = expand_votes(Fh, Yh)
    sk = LogisticRegression(C=C, max_iter=10000, tol=1e-10).fit(Xe, ye, sample_weight=we)
    W, b, _ = M.fit_head(Fh, Yh, C, device='cpu', max_iter=2000)
    diff = np.abs(M.head_proba(W, b, Fh) - sk.predict_proba(Fh)).max()
    print('2. head vs scikit-learn, C = %g: max |difference in probabilities| %.5f' % (C, diff))
    assert diff < 2e-3, 'PyTorch head differs from scikit-learn'
print('head checks passed')

try:
    from aeon.transformations.collection.convolution_based import MiniRocket
except ImportError:
    print('aeon not installed: checks 3-5 skipped')
    sys.exit(0)

# a small EEG-like set: 16 channels x 1000 points, slow rhythms + noise, varied amplitude
n, C, T = 24, 16, 1000
t = np.arange(T) / 100
X = (rng.normal(size=(n, C, 1)) * np.sin(2 * np.pi * rng.uniform(0.5, 8, (n, C, 1)) * t)
     + 0.5 * rng.normal(size=(n, C, T))).astype(np.float32)

ref = MiniRocket(n_kernels=10_000, random_state=0).fit(X)
nc, ci, dil, per, bias = ref.parameters
F_ref = ref.transform(X)

# 3 ------------------------------------------------------------------------------------------
params = dict(dilations=dil, per_dilation=per, n_channels_per_comb=nc, channel_indices=ci, biases=bias)
F_ours = M.transform(X, params, device='cpu', batch=8)
diff = np.abs(F_ours - F_ref)
print('3. transform vs aeon: %d features | max |diff| %.4f | entries differing %.5f%% (one time point = %.4f)'
      % (F_ref.shape[1], diff.max(), 100 * (diff > 0).mean(), 1 / T))
assert F_ours.shape == F_ref.shape
assert (diff > 0).mean() < 1e-3 and diff.max() <= 0.01, 'transform differs from aeon'   # rounding at ties only

# 4 ------------------------------------------------------------------------------------------
ours = M.fit_params(X, seed=0)
assert np.array_equal(ours['n_channels_per_comb'], nc) and np.array_equal(ours['channel_indices'], ci)
assert np.array_equal(ours['dilations'], dil) and np.array_equal(ours['per_dilation'], per)
print('4. channel subsets, dilations and features per dilation: identical to aeon for seed 0')

# 5 ------------------------------------------------------------------------------------------
q = M.quantiles(84 * int(per.sum()))
f = start_c = comb = 0
checked = matched = 0
for d, nf in zip(dil, per):
    for k in range(84):
        chans = ci[start_c:start_c + nc[comb]]
        if comb % 7 == 0:                     # a sample of combinations keeps the test fast
            cands = [np.quantile(M.response_np(X[i][chans], int(d), M.KERNELS[k]), q[f:f + nf]) for i in range(n)]
            checked += 1
            matched += any(np.allclose(c, bias[f:f + nf], rtol=1e-4, atol=1e-4) for c in cands)
        start_c += nc[comb]
        f += nf
        comb += 1
print('5. biases: %d of %d sampled combinations equal the quantiles of our response on a training window'
      % (matched, checked))
assert matched == checked, 'bias computation differs from aeon'
print('all MiniRocket checks passed')
