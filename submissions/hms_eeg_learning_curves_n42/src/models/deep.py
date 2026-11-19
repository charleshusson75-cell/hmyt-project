"""One training recipe for every deep model (PyTorch), so no network gets a hand-made recipe.

    val_p, test_p, info = train_predict(net, split, seed, lr=...)

Recipe 2 (10 November 2026): the training length and the patience are counted in gradient steps, not
epochs, so every training size gets the same optimisation. Recipe 1 counted epochs and stopped after
about 45 gradient steps at N = 100, before the CNN even reached the mean-vote baseline.

- Loss: cross-entropy against the vote shares (= KL + a constant), the objective of every model.
- AdamW (weight decay 0.01), batch 64, linear warm-up over 5% of the steps then cosine decay.
- Length: at most 6,000 gradient steps, but at least 30 epochs for large N (STEPS, MIN_EPOCHS).
- Early stopping: KL on the high-quality validation windows every 50 steps (every epoch if an epoch
  is longer); stop when it has not improved for 2,000 steps, never before 2,000 steps; the best
  weights are kept.
- Learning rate: the only setting chosen per model, on the validation set at N = 1,000, over 3 seeds
  (the curve notebook does it once, from LR_GRID); not tuned per N.
- Optional second stage (finetune_hq=True): from the best weights, train only on the training windows
  with >= 10 votes, learning rate x 0.1, at most 1,000 steps, patience 500 steps; its weights are kept
  only if the validation KL improves. Same recordings and windows: only the weighting of the labels changes.
- Reproducibility: seeded data order and initialisation, deterministic cuDNN kernels. v1's
  TensorFlow CNN differed by up to 0.12 KL between two GPU runs with the same seed.
- Predictions (19 November 2026): a window whose float16 logits overflow is predicted again in float32
  (`fp32_rows` in info counts them). No run finished before that date had such a window, so its results
  do not change; the likely cause of 4 CNN runs failing on every attempt (decision D-56).
"""
import copy, math, os, time
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # needed for deterministic cuBLAS; before CUDA starts

RECIPE = 2
LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3)
BATCH, WEIGHT_DECAY, WARMUP = 64, 0.01, 0.05
STEPS, MIN_EPOCHS, EVAL_STEPS = 6000, 30, 50
PATIENCE_STEPS, MIN_STEPS = 2000, 2000
FT_LR_FACTOR, FT_STEPS, FT_PATIENCE = 0.1, 1000, 500


def budget(n, batch=BATCH, steps=STEPS, min_epochs=MIN_EPOCHS):
    """(steps per epoch, maximum number of gradient steps) for n training windows."""
    spe = math.ceil(n / batch)
    return spe, max(steps, min_epochs * spe)


def seed_everything(seed):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _predict(net, X, device, batch=256, stats=None):
    """Class probabilities. On the GPU the forward pass runs in float16; a window whose logits overflow
    float16 (probabilities NaN) is computed again in float32, so one extreme window cannot void a run.
    Rows that were already finite are unchanged. stats (a dict), if given, counts the recomputed rows."""
    import torch
    p = _predict16(net, X, device, batch)
    bad = ~np.isfinite(p).all(1)
    if bad.any():
        idx = np.flatnonzero(bad)
        with torch.no_grad():
            for i in range(0, len(idx), batch):
                j = idx[i:i + batch]
                x = torch.as_tensor(X[j], device=device)
                p[j] = torch.softmax(net(x).float(), 1).double().cpu().numpy()
        if stats is not None:
            stats['fp32_rows'] = stats.get('fp32_rows', 0) + int(bad.sum())
    return p


def _predict16(net, X, device, batch=256):
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


def _stage(net, Xtr, Ytr, Xva, Yva, device, lr, max_steps, patience, min_steps, seed, batch, weight_decay,
           best=None, verbose=False):
    """Train `net` from its current weights. best = (state, validation KL) to beat, or None.
    Returns (best state, best validation KL, history)."""
    import torch
    from hms_eval import kl_per_sample
    n = len(Xtr)
    spe = math.ceil(n / batch)
    every = max(1, math.ceil(EVAL_STEPS / spe))       # evaluate every `every` epochs (>= 50 steps)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    warm = max(1, int(WARMUP * max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (t - warm) / max(1, max_steps - warm)))))
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    gen = torch.Generator().manual_seed(seed)
    Xtr_t, Ytr_t = torch.as_tensor(Xtr), torch.as_tensor(Ytr, dtype=torch.float32)
    best_state, best_kl = best if best else (None, np.inf)
    best_step, step, epoch, tot, seen = 0, 0, 0, 0.0, 0
    hist = dict(step=[], val_kl=[], train_loss=[])
    while step < max_steps:
        epoch += 1
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            if step >= max_steps:
                break
            idx = perm[i:i + batch]
            x, y = Xtr_t[idx].to(device, non_blocking=True), Ytr_t[idx].to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                logits = net(x)
            loss = -(y * torch.log_softmax(logits.float(), 1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tot += loss.item() * len(idx)
            seen += len(idx)
        if epoch % every and step < max_steps:
            continue
        v = float(kl_per_sample(Yva, _predict(net, Xva, device)).mean())
        hist['step'].append(step)
        hist['val_kl'].append(round(v, 5))
        hist['train_loss'].append(round(tot / max(seen, 1), 5))
        tot, seen = 0.0, 0
        if verbose:
            print('  step %5d  train %.4f  val KL %.4f' % (step, hist['train_loss'][-1], v), flush=True)
        if v < best_kl - 1e-6:
            best_state, best_kl, best_step = copy.deepcopy(net.state_dict()), v, step
        elif step - best_step >= patience and step >= min_steps:
            break
    return best_state, best_kl, dict(hist, steps=step, epochs=epoch, best_step=best_step, steps_per_epoch=spe)


def train_predict(net, split, seed=0, lr=1e-3, max_steps=None, patience_steps=PATIENCE_STEPS, min_steps=MIN_STEPS,
                  batch=BATCH, weight_decay=WEIGHT_DECAY, to_input=None, finetune_hq=False, ft_steps=FT_STEPS,
                  ft_patience=FT_PATIENCE, verbose=False):
    """net: a torch module taking a float32 batch shaped by `to_input` (default: (n, channels, time))
    and returning logits for the 6 classes."""
    import torch
    seed_everything(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    to_input = to_input or (lambda a: np.ascontiguousarray(np.transpose(np.asarray(a, np.float32), (0, 2, 1))))
    Xtr, Xva, Xte = to_input(split.X_train), to_input(split.X_val), to_input(split.X_test)
    sel = np.asarray(getattr(split, 'val_hq', np.ones(len(split.Y_val), bool)))
    Xva_hq, Yva_hq = Xva[sel], split.Y_val[sel]
    spe, steps_max = budget(len(Xtr), batch)
    max_steps = max_steps or steps_max
    net = net.to(device)
    t0 = time.time()
    state, kl1, h = _stage(net, Xtr, split.Y_train, Xva_hq, Yva_hq, device, lr, max_steps, patience_steps,
                           min(min_steps, max_steps), seed, batch, weight_decay, verbose=verbose)
    info = dict(recipe=RECIPE, lr=lr, batch=batch, weight_decay=weight_decay, max_steps=max_steps,
                patience_steps=patience_steps, steps=h['steps'], best_step=h['best_step'],
                epochs=h['epochs'], best_epoch=math.ceil(h['best_step'] / spe), steps_per_epoch=spe,
                eval_steps=h['step'], val_kl_per_eval=h['val_kl'], train_loss_per_eval=h['train_loss'],
                val_kl_stage1=round(kl1, 5))
    if finetune_hq:
        hq = np.asarray(split.hq_train, bool)
        info['ft_windows'] = int(hq.sum())
        if hq.any():
            net.load_state_dict(state)
            state, kl2, h2 = _stage(net, Xtr[hq], split.Y_train[hq], Xva_hq, Yva_hq, device, lr * FT_LR_FACTOR,
                                    ft_steps, ft_patience, 0, seed + 1, batch, weight_decay, best=(state, kl1),
                                    verbose=verbose)
            info.update(ft_lr=lr * FT_LR_FACTOR, ft_max_steps=ft_steps, ft_steps=h2['steps'],
                        ft_best_step=h2['best_step'], ft_improved=h2['best_step'] > 0,
                        ft_eval_steps=h2['step'], ft_val_kl_per_eval=h2['val_kl'], val_kl_after_ft=round(kl2, 5))
    net.load_state_dict(state)
    st = {}
    val_p, test_p = _predict(net, Xva, device, stats=st), _predict(net, Xte, device, stats=st)
    info.update(params=int(sum(p.numel() for p in net.parameters())), train_seconds=round(time.time() - t0, 1),
                device=device, torch=torch.__version__, deterministic=True, fp32_rows=st.get('fp32_rows', 0))
    return val_p, test_p, info
