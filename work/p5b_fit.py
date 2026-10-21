"""Fit ONE model on ONE training tier and save the predicted val fingerprints.

Usage: python work/p5b_fit.py <ridge|mlp|deep> <A25|A50|A|B|D> [artifacts dir, default artifacts/scale]
  ridge : NormalEqRidge alpha 1 (v0.2c settings)
  mlp   : 1024x1024, dropout 0.5 (the P5 'official' configuration, chosen by inner-holdout BCE)
  deep  : 1024 -> 4 residual blocks (LayerNorm) -> 1024 outputs, dropout 0.3
-> <dir>/pred_<model>_<tier>.npy, <dir>/hist_<model>_<tier>.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "src")
model, tier = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "artifacts/scale"
R = pd.read_parquet(f"{OUT}/rows_train.parquet")
m = R[f"in_{tier}"].values
X = sp.load_npz(f"{OUT}/X_train.npz").tocsr()[m].astype(np.float32).tocsr()
Xva = sp.load_npz(f"{OUT}/X_val.npz").astype(np.float32).tocsr()
Y = np.load(f"{OUT}/mol_fp.npy")[R.mol_idx.values[m]]                      # uint8, one row per spectrum
print(f"{model} on tier {tier}: {X.shape[0]:,} spectra, {R[m].key.nunique():,} molecules, {X.shape[1]:,} columns", flush=True)
t0 = time.time(); info = {}
if model == "ridge":
    import v1_pipeline as v1
    reg = v1.NormalEqRidge(alpha=1.0).fit(X, Y)
    P = reg.predict(Xva).astype(np.float32)
else:
    import mlp
    keep = np.flatnonzero(np.asarray((X != 0).sum(0)).ravel() > 0); X, Xva = X[:, keep], Xva[:, keep]
    cfg = dict(hidden=(1024, 1024), dropout=0.5, blocks=0) if model == "mlp" else dict(hidden=(1024,), dropout=0.3, blocks=4)
    max_epochs = 20 if (model == "deep" and tier == "D") else 60          # time cap for the depth probe on the biggest tier (documented in the results)
    net, info = mlp.fit_mlp(X, Y, R.scaffold.values[m].astype(str), max_epochs=max_epochs, log=lambda s: print(s, flush=True), **cfg)
    P = mlp.predict(net, Xva, next(net.parameters()).device).astype(np.float32)
np.save(f"{OUT}/pred_{model}_{tier}.npy", P)
info.update(model=model, tier=tier, n_spectra=int(X.shape[0]), n_molecules=int(R[m].key.nunique()), seconds=time.time() - t0)
json.dump(info, open(f"{OUT}/hist_{model}_{tier}.json", "w"))
print(f"saved pred_{model}_{tier}.npy | {info['seconds'] / 60:.1f} min", flush=True)
