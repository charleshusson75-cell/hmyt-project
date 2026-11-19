"""Explainability of the main models (notebook section 11), one part per process: LightGBM and PyTorch in the
same process crash or hang on macOS (two OpenMP runtimes, decision D-46), so each part is its own process.

    python xai_run.py --part lgbm  --data <prepared dir> --results <results dir> --out <figures dir>
    python xai_run.py --part torch --data <prepared dir> --results <results dir> --out <figures dir> --kw lr=0.001

Fold 0, all training recordings, seed 0, the windows of the test set with >= 10 votes.
- lgbm:  LightGBM's SHAP values, as shares by frequency band and by chain group.
- torch: the logistic regression on band power (its coefficients, refitted with the C of the saved run), and the
         CNN + fine-tuning, retrained with the same seed (its test predictions are compared with the saved run),
         then channel occlusion, Grad-CAM on one clear test window per class (xai_cnn_gradcam.png) and on
         up to 6 misclassified windows (xai_cnn_gradcam_errors.png: a missed seizure, a false seizure, then
         the most frequent confusions); the cases are listed in xai_cnn_cases.csv.
--kw key=value goes to models.deep.train_predict (e.g. lr=0.001).
"""
import argparse, ast, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _save(fig_or_none, out, name):
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(out, name), dpi=150, bbox_inches='tight')
    plt.close('all')
    print('saved', name, flush=True)


def part_lgbm(a):
    import numpy as np
    import hms_data, hms_xai as HX
    from models import lgbm
    sp = hms_data.make_split(hms_data.load(a.data), None, 0, signals=False)
    hq = np.flatnonzero(sp.test_hq)
    clf, _ = lgbm.fit(sp, 0)
    imp = np.abs(HX.lgbm_shap(clf, sp.F_test[hq])).mean(0)        # (classes, features): mean |SHAP|
    tb, tc = HX.by_band_and_chain(imp)
    for t, name in ((tb, 'band'), (tc, 'chains')):
        t.to_csv(os.path.join(a.out, 'xai_lgbm_shap_by_%s.csv' % name))
        HX.plot_heatmap(t, 'LightGBM: share of mean |SHAP| by %s' % name)
        _save(None, a.out, 'xai_lgbm_shap_by_%s.png' % name)
    print(tb.to_string())


def part_torch(a, kw):
    import numpy as np
    import torch
    import hms_data, hms_xai as HX
    from models import cnn1d, deep
    from models.minirocket import fit_head, standardise
    sp = hms_data.make_split(hms_data.load(a.data), None, 0)
    hq = np.flatnonzero(sp.test_hq)
    # logistic regression on band power, with the C its saved run chose on validation
    C = json.load(open(os.path.join(a.results, 'logreg_band_n%d_s0.json' % sp.n_train)))['info']['C']
    Ftr, _ = standardise(np.asarray(sp.F_train, np.float32), np.asarray(sp.F_test, np.float32))
    W = fit_head(Ftr, sp.Y_train, C)[0].cpu().numpy().T          # (classes, features)
    HX.top_features(W).to_csv(os.path.join(a.out, 'xai_logreg_top_features.csv'), index=False)
    print(HX.top_features(W).to_string())
    tb, tc = HX.by_band_and_chain(np.abs(W))
    for t, name in ((tb, 'band'), (tc, 'chains')):
        t.to_csv(os.path.join(a.out, 'xai_logreg_by_%s.csv' % name))
        HX.plot_heatmap(t, 'Logistic regression: share of |coefficients| by %s' % name)
        _save(None, a.out, 'xai_logreg_by_%s.png' % name)
    # CNN + fine-tuning: the scored model, retrained with the same seed
    deep.seed_everything(0)
    net = cnn1d._net(sp.X_train.shape[2], sp.Y_train.shape[1])
    _, tp, _ = deep.train_predict(net, sp, seed=0, finetune_hq=True, **kw)
    saved = np.load(os.path.join(a.results, 'preds', 'cnn1d_ft_n%d_s0_test.npy' % sp.n_train))
    print('same model as the scored run: max |difference| in test probabilities %.2e' % np.abs(tp - saved).max(), flush=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    occ = HX.channel_occlusion(net, sp.X_test[hq], sp.Y_test[hq], dev)
    occ.to_csv(os.path.join(a.out, 'xai_cnn_occlusion.csv'))
    HX.plot_heatmap(occ.clip(lower=0), "CNN + fine-tuning: drop in the probability of the experts' class when a chain is removed",
                    fmt='%.3f', cmap='Oranges')
    _save(None, a.out, 'xai_cnn_occlusion.png')
    import matplotlib.pyplot as plt
    import pandas as pd
    C = hms_data.CLASSES
    ex = HX.examples(sp.Y_test[hq], tp[hq])
    cases = []

    def case(kind, i, j):                          # one row of xai_cnn_cases.csv: what the experts and the model said
        cases.append(dict(kind=kind, eeg_id=int(sp.ids_test[i]), eeg_sub_id=int(sp.subids_test[i]),
                          votes=int(sp.total_votes_test[i]), experts=C[sp.Y_test[i].argmax()],
                          experts_share=round(float(sp.Y_test[i].max()), 3), model=C[tp[i].argmax()],
                          model_prob=round(float(tp[i].max()), 3), gradcam_class=C[j],
                          experts_shares=' '.join('%s %.2f' % (c, v) for c, v in zip(C, sp.Y_test[i]) if v >= 0.05),
                          model_probs=' '.join('%s %.2f' % (c, v) for c, v in zip(C, tp[i]) if v >= 0.05)))
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    for ax, (cls, idx) in zip(axes.ravel(), ex.items()):
        i, j = hq[idx[0]], C.index(cls)
        HX.plot_gradcam(sp.X_test[i], HX.gradcam(net, sp.X_test[i], j, dev),
                        '%s: experts %.0f%%, model %.0f%%' % (cls, 100 * sp.Y_test[i, j], 100 * tp[i, j]), ax=ax)
        case('correct', i, j)
    for ax in axes.ravel()[len(ex):]:
        ax.axis('off')
    plt.tight_layout()
    _save(None, a.out, 'xai_cnn_gradcam.png')
    # misclassified windows: Grad-CAM for the class the model chose, i.e. what pushed it to the wrong answer
    err = HX.error_examples(sp.Y_test[hq], tp[hq])
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    for ax, idx in zip(axes.ravel(), err):
        i = hq[idx]
        t, j = sp.Y_test[i].argmax(), tp[i].argmax()
        HX.plot_gradcam(sp.X_test[i], HX.gradcam(net, sp.X_test[i], j, dev),
                        'experts: %s %.0f%%  |  model: %s %.0f%% (Grad-CAM for %s)'
                        % (C[t], 100 * sp.Y_test[i, t], C[j], 100 * tp[i, j], C[j]), ax=ax)
        case('error', i, j)
    for ax in axes.ravel()[len(err):]:
        ax.axis('off')
    plt.tight_layout()
    _save(None, a.out, 'xai_cnn_gradcam_errors.png')
    pd.DataFrame(cases).to_csv(os.path.join(a.out, 'xai_cnn_cases.csv'), index=False)
    print(pd.DataFrame(cases).to_string(), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', required=True, choices=['lgbm', 'torch'])
    ap.add_argument('--data', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--kw', action='append', default=[])
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    os.makedirs(a.out, exist_ok=True)
    kw = {}
    for s in a.kw:
        k, v = s.split('=', 1)
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
    part_lgbm(a) if a.part == 'lgbm' else part_torch(a, kw)


if __name__ == '__main__':
    main()
