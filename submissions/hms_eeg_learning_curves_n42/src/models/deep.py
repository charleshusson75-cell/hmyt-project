"""One training recipe for every deep model (PyTorch), so no network gets a hand-made recipe.

    val_p, test_p, info = train_predict(net, split, seed, lr=...)

- Loss: cross-entropy against the vote shares (= KL + a constant), the objective of every model.
- AdamW (weight decay 0.01), batch 64, linear warm-up over 5% of the steps then cosine decay.
- Training length: about 6,000 gradient steps, at least 30 and at most 200 epochs (EPOCH_RULE);
  early stopping when the KL on the high-quality validation windows has not improved for 15 epochs;
  the best epoch's weights are kept.
- Learning rate: the only setting chosen per model, on the validation set at N = 1,000
  (the curve notebook does it once, from LR_GRID); not tuned per N.
- Reproducibility: seeded data order and initialisation, deterministic cuDNN kernels. v1's
  TensorFlow CNN differed by up to 0.12 KL between two GPU runs with the same seed.
"""
import copy, math, os, time
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # needed for deterministic cuBLAS; before CUDA starts

LR_GRID = (3e-4, 1e-3, 3e-3)
BATCH, WEIGHT_DECAY, WARMUP, PATIENCE = 64, 0.01, 0.05, 15


def epochs_for(n, batch=BATCH, steps=6000, lo=30, hi=200):
    return int(np.clip(round(steps / math.ceil(n / batch)), lo, hi))


def seed_everything(seed):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _predict(net, X, device, batch=256):
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.as_tensor(X[i:i + batch], device=device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                logits = net(x)
            out.append(torch.softmax(logits.float(), 1).double().cpu().numpy())
    return np.concatenate(out)


def train_predict(net, split, seed=0, lr=1e-3, max_epochs=None, patience=PATIENCE, batch=BATCH,
                  weight_decay=WEIGHT_DECAY, to_input=None, verbose=False):
    """net: a torch module taking a float32 batch shaped by `to_input` (default: (n, channels, time))
    and returning logits for the 6 classes."""
    import torch
    from hms_eval import kl_per_sample
    seed_everything(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    to_input = to_input or (lambda a: np.ascontiguousarray(np.transpose(np.asarray(a, np.float32), (0, 2, 1))))
    Xtr, Xva, Xte = to_input(split.X_train), to_input(split.X_val), to_input(split.X_test)
    Ytr = torch.as_tensor(split.Y_train, dtype=torch.float32)
    sel = getattr(split, 'val_hq', np.ones(len(split.Y_val), bool))
    n = len(Xtr)
    max_epochs = max_epochs or epochs_for(n, batch)
    steps_per_epoch = math.ceil(n / batch)
    total = max_epochs * steps_per_epoch
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    warm = max(1, int(WARMUP * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    gen = torch.Generator().manual_seed(seed)
    Xtr_t = torch.as_tensor(Xtr)
    best, best_kl, best_epoch, wait = None, np.inf, 0, 0
    val_kl, train_loss, t0 = [], [], time.time()
    for epoch in range(1, max_epochs + 1):
        net.train()
        perm, tot = torch.randperm(n, generator=gen), 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            x, y = Xtr_t[idx].to(device, non_blocking=True), Ytr[idx].to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                logits = net(x)
            loss = -(y * torch.log_softmax(logits.float(), 1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(idx)
        pv = _predict(net, Xva, device)
        v = float(kl_per_sample(split.Y_val[sel], pv[sel]).mean())
        val_kl.append(round(v, 5))
        train_loss.append(round(tot / n, 5))
        if verbose:
            print('  epoch %3d  train %.4f  val KL %.4f' % (epoch, train_loss[-1], v), flush=True)
        if v < best_kl - 1e-6:
            best, best_kl, best_epoch, wait = copy.deepcopy(net.state_dict()), v, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best)
    val_p, test_p = _predict(net, Xva, device), _predict(net, Xte, device)
    info = dict(lr=lr, batch=batch, weight_decay=weight_decay, max_epochs=max_epochs, epochs=len(val_kl),
                best_epoch=best_epoch, val_kl_per_epoch=val_kl, train_loss_per_epoch=train_loss,
                params=int(sum(p.numel() for p in net.parameters())), train_seconds=round(time.time() - t0, 1),
                device=device, torch=torch.__version__, deterministic=True)
    return val_p, test_p, info
