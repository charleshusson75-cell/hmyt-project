"""1D CNN trained from scratch on the 16-channel signal at 200 Hz: a ResNet-18-sized xresnet1d.

Architecture: the "bag of tricks" ResNet (xresnet: three-convolution stem, average-pooling
shortcuts, last batch-norm of each block initialised at zero) in its 1D form, as in the PTB-XL
benchmark of Strodthoff et al. (2021): kernel size 5, concat pooling (average + max), a hidden
layer of 128 units with batch-norm and dropout. Depth 18 ([2, 2, 2, 2] blocks), the standard
small configuration. Training: the shared recipe of models/deep.py, learning rate chosen on the
validation set.

This replaces v1's TensorFlow CNN, whose sampling rate (100 Hz) and recipe came from the ECG work.
"""
NAME = 'cnn1d'
INPUT = 'signal'


def _net(c_in, n_out, layers=(2, 2, 2, 2), ks=5, stem=(32, 32, 64), widths=(64, 128, 256, 512), p=0.5):
    import torch
    import torch.nn as nn

    def conv(ni, nf, k, stride=1, act=True, zero_bn=False):
        bn = nn.BatchNorm1d(nf)
        nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
        mods = [nn.Conv1d(ni, nf, k, stride=stride, padding=k // 2, bias=False), bn]
        return nn.Sequential(*(mods + [nn.ReLU(inplace=True)] if act else mods))

    class Block(nn.Module):
        def __init__(self, ni, nf, stride):
            super().__init__()
            self.convs = nn.Sequential(conv(ni, nf, ks, stride), conv(nf, nf, ks, act=False, zero_bn=True))
            idp = ([nn.AvgPool1d(2, ceil_mode=True)] if stride != 1 else []) + ([conv(ni, nf, 1, act=False)] if ni != nf else [])
            self.idpath = nn.Sequential(*idp)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.convs(x) + self.idpath(x))

    class ConcatPool(nn.Module):                     # average and max over time, side by side
        def forward(self, x):
            return torch.cat([nn.functional.adaptive_avg_pool1d(x, 1),
                              nn.functional.adaptive_max_pool1d(x, 1)], 1).flatten(1)

    mods = [conv(c_in, stem[0], ks, stride=2), conv(stem[0], stem[1], ks), conv(stem[1], stem[2], ks),
            nn.MaxPool1d(3, stride=2, padding=1)]
    ni = stem[2]
    for i, (w, nb) in enumerate(zip(widths, layers)):
        for j in range(nb):
            mods.append(Block(ni, w, 2 if (j == 0 and i > 0) else 1))
            ni = w
    head = [ConcatPool(), nn.BatchNorm1d(2 * ni), nn.Dropout(p / 2), nn.Linear(2 * ni, 128), nn.ReLU(inplace=True),
            nn.BatchNorm1d(128), nn.Dropout(p), nn.Linear(128, n_out)]
    net = nn.Sequential(*mods, *head)
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    return net


def fit_predict(split, seed=0, lr=1e-3, **kw):
    from models.deep import train_predict, seed_everything
    seed_everything(seed)                            # same initial weights for a given seed
    net = _net(split.X_train.shape[2], split.Y_train.shape[1])
    val_p, test_p, info = train_predict(net, split, seed=seed, lr=lr, **kw)
    info.update(arch='xresnet1d18', kernel=5, sampling_rate=200)
    return val_p, test_p, info
