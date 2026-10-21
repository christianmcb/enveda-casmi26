"""P5 step 0: build the v0.2c features ONCE and cache them (sparse, empty train columns dropped) so MLP runs start in seconds.

Run from the repo root:  python work/p5_prepare.py
-> artifacts/feat_v02c_tr.npz, feat_v02c_va.npz, ytr.npy, tr_scaffold.npy
"""
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "src")
import v1_pipeline as v1

D = v1.load_data()
feat = v1.Features(D["tr"], neutral_loss=True, frag_defect=0.02, loss_defect=0.02)
Xtr, Xva = feat.build(D["tr"], D["mz_t"], D["it_t"]), feat.build(D["va"], D["mz_v"], D["it_v"])
keep = np.flatnonzero(np.asarray((Xtr != 0).sum(0)).ravel() > 0)
sp.save_npz("artifacts/feat_v02c_tr.npz", Xtr[:, keep].tocsr()); sp.save_npz("artifacts/feat_v02c_va.npz", Xva[:, keep].tocsr())
np.save("artifacts/ytr.npy", D["Ytr"].astype(np.float32)); np.save("artifacts/tr_scaffold.npy", D["tr"].scaffold.values.astype(str))
print("saved", Xtr[:, keep].shape, Xva[:, keep].shape, "nnz/row", Xtr[:, keep].nnz / Xtr.shape[0])
