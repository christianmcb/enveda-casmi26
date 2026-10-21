"""P5: an MLP that predicts the 1,024 Morgan bits from sparse spectrum features (BCE loss).

Early stopping uses a scaffold-grouped holdout carved out of the TRAINING data, so the validation set is never used for any
training decision (hyper-parameters are chosen by this inner holdout loss only).
"""
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit


class ResBlock(nn.Module):
    """Pre-norm residual block: x + MLP(LayerNorm(x))."""
    def __init__(self, d, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x):
        return x + self.f(self.norm(x))


def make_net(n_in, hidden=(1024, 1024), dropout=0.3, n_out=1024, blocks=0):
    """blocks > 0: sparse input -> Linear(hidden[0]) -> `blocks` residual blocks -> LayerNorm -> Linear(n_out)  (deeper variant)."""
    if blocks:
        d = hidden[0]
        return nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Dropout(dropout), *[ResBlock(d, dropout) for _ in range(blocks)],
                             nn.LayerNorm(d), nn.Linear(d, n_out))
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, n_out))


def _dense(X, idx, device):
    return torch.from_numpy(X[idx].toarray()).to(device)


def predict(net, X, device, bs=1024):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            out.append(torch.sigmoid(net(_dense(X, np.arange(i, min(i + bs, X.shape[0])), device))).cpu().numpy())
    return np.vstack(out)


def fit_mlp(X, Y, scaffolds, hidden=(1024, 1024), dropout=0.3, lr=1e-3, weight_decay=1e-2, batch_size=256,
            max_epochs=60, patience=6, seed=42, device=None, log=print, blocks=0):
    """X: scipy CSR float32 (n x p); Y: (n x 1024) 0/1 as uint8 or float32 (converted per batch on the device). Returns (net, history)."""
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    fit_i, es_i = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(X, groups=scaffolds))
    Xf, Xe = X[fit_i], X[es_i]
    Yf, Ye = torch.from_numpy(Y[fit_i]).to(device), torch.from_numpy(Y[es_i]).to(device)
    net = make_net(X.shape[1], hidden, dropout, blocks=blocks).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad, hist = 1e9, None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(max_epochs):
        t0 = time.time(); net.train(); tr_loss = 0.0
        perm = rng.permutation(len(fit_i))
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossf(net(_dense(Xf, b, device)), Yf[torch.from_numpy(b).to(device)].float())
            loss.backward(); opt.step(); tr_loss += loss.item() * len(b)
        net.eval(); es = 0.0
        with torch.no_grad():
            for i in range(0, Xe.shape[0], 1024):
                j = np.arange(i, min(i + 1024, Xe.shape[0]))
                es += lossf(net(_dense(Xe, j, device)), Ye[torch.from_numpy(j).to(device)].float()).item() * len(j)
        es /= Xe.shape[0]; tr_loss /= len(perm)
        hist.append(dict(epoch=ep, train_bce=tr_loss, holdout_bce=es, seconds=time.time() - t0))
        log(f"  epoch {ep:2d} | train BCE {tr_loss:.5f} | inner-holdout BCE {es:.5f} | {time.time() - t0:.0f}s")
        if es < best - 1e-5:
            best, bad, best_state = es, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, dict(history=hist, best_holdout_bce=best, best_epoch=int(np.argmin([h["holdout_bce"] for h in hist])))
