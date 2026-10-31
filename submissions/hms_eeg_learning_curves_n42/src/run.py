"""Run ONE model on ONE training size with ONE seed, score it, save one result.

    python run.py --model cnn1d --n 1000 --seed 0 --data <prepared dir> --out results
    python run.py --model floor --n all  --seed 0 --data <prepared dir> --out results
    python run.py --model cnn1d --n 100  --seed 1 --data <prepared dir> --kw lr=0.003

--skip-existing makes a loop resumable: runs whose result file exists are not redone.

The learning curve is a loop over this script. Only the v1 model xresnet needs p2lib.py next to
this file (or in the folder given by the P2LIB_DIR environment variable).
"""
import argparse, ast, importlib, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get('P2LIB_DIR'):
    sys.path.insert(0, os.environ['P2LIB_DIR'])


def _value(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='a file in models/, e.g. xresnet or floor')
    ap.add_argument('--n', required=True, help='number of training windows, or "all"')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data', required=True, help='folder written by prepare.py')
    ap.add_argument('--out', default='results')
    ap.add_argument('--kw', action='append', default=[], help='model argument key=value (repeatable)')
    ap.add_argument('--skip-existing', action='store_true', help='do nothing if this result already exists')
    a = ap.parse_args(argv)

    model = importlib.import_module('models.' + a.model)
    if hasattr(model, 'setup'):
        model.setup()                              # before pandas is imported: see models/xresnet.py
    import hms_data, hms_eval

    kw = {k: _value(v) for k, v in (s.split('=', 1) for s in a.kw)}
    n = None if a.n == 'all' else int(a.n)
    data = hms_data.load(a.data)
    pool = int((data.meta['split'] == 'train').sum())
    n_eff = pool if n is None or n >= pool else n
    tag = '%s_n%d_s%d' % (model.NAME, n_eff, a.seed)
    if n_eff > getattr(model, 'MAX_N', n_eff):     # e.g. TabPFN: pretrained for up to 10,000 samples
        print('skip: %s is limited to N <= %d (asked %d)' % (model.NAME, model.MAX_N, n_eff), flush=True)
        return None
    if a.skip_existing and os.path.exists(os.path.join(a.out, tag + '.json')):
        print('skip (already done):', tag, flush=True)
        return None
    split = hms_data.make_split(data, n, a.seed, signals=(model.INPUT == 'signal'))
    t0 = time.time()
    val_p, test_p, info = model.fit_predict(split, seed=a.seed, **kw)
    info['seconds'] = round(time.time() - t0, 1)
    res = hms_eval.score(split, val_p, test_p)
    path = hms_eval.save_result(a.out, model.NAME, split, res, info, val_p, test_p)
    print(hms_eval.one_line(model.NAME, res), '|', '%.0f s' % info['seconds'], '->', path, flush=True)
    del split, val_p, test_p                       # free memory before the next run of a long loop
    import gc
    gc.collect()
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res


if __name__ == '__main__':
    main()
