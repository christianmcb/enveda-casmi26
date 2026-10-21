"""Fit ONE configuration in its own process and save the predicted val fingerprints (keeps peak memory bounded).

Usage: python work/p4c_fit.py <name> <alpha> <frag_defect> <loss_defect>      e.g.  python work/p4c_fit.py v0.2c 1 0.02 0.02
-> artifacts/p4c_pred_<name>.npy (14,547 x 1,024 float32)
"""
import sys

import numpy as np

sys.path.insert(0, "src")
import v1_pipeline as v1

name, alpha, fd, ld = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
D = v1.load_data()
M = v1.fit_ridge(D, alpha=alpha, neutral_loss=True, solver="normal_eq", frag_defect=fd, loss_defect=ld)
np.save(f"artifacts/p4c_pred_{name}.npy", M["reg"].predict(M["Xva"]).astype(np.float32))
print(f"saved artifacts/p4c_pred_{name}.npy | columns {M['Xtr'].shape[1]:,}", flush=True)
