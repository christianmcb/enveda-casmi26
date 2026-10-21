"""Fit ONE MLP configuration on the cached v0.2c features and save the predicted val fingerprints.

Usage: python work/p5_fit_mlp.py <name> <hidden e.g. 1024,1024> <dropout> [max_epochs]
-> artifacts/p5_pred_<name>.npy, artifacts/p5_hist_<name>.json
"""
import json
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "src")
import mlp

name, hidden, dropout = sys.argv[1], tuple(int(h) for h in sys.argv[2].split(",")), float(sys.argv[3])
max_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
X, Xva = sp.load_npz("artifacts/feat_v02c_tr.npz").astype(np.float32).tocsr(), sp.load_npz("artifacts/feat_v02c_va.npz").astype(np.float32).tocsr()
Y, scaf = np.load("artifacts/ytr.npy"), np.load("artifacts/tr_scaffold.npy", allow_pickle=True)
print(f"{name}: X {X.shape}, hidden {hidden}, dropout {dropout}", flush=True)
net, info = mlp.fit_mlp(X, Y, scaf, hidden=hidden, dropout=dropout, max_epochs=max_epochs, log=lambda s: print(s, flush=True))
np.save(f"artifacts/p5_pred_{name}.npy", mlp.predict(net, Xva, next(net.parameters()).device).astype(np.float32))
json.dump(info, open(f"artifacts/p5_hist_{name}.json", "w"))
print(f"saved p5_pred_{name}.npy | best inner-holdout BCE {info['best_holdout_bce']:.5f} at epoch {info['best_epoch']}", flush=True)
