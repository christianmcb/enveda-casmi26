"""P4a: add neutral-loss features (precursor m/z - fragment m/z, 1 Da bins). Single change vs v0.1b.

Run from the repo root:  python work/p4a_neutral_loss.py     -> artifacts/p4a_results.csv
Both versions are scored in the same run on the same pool/slices, so the comparison is exactly attributable.
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

D = v1.load_data()
B = bm.Bench(D)
print("train-only offsets (ppm):", B.BIAS)
res = []
for name, nl in [("v0.1b", False), ("v0.2a_neutral_loss", True)]:
    M = v1.fit_ridge(D, neutral_loss=nl)
    print(name, "features:", M["Xtr"].shape, flush=True)
    res.append(B.score(name, M["reg"].predict(M["Xva"])))
    print(res[-1].round(4).to_string(index=False), flush=True)
out = pd.concat(res)
out.to_csv("artifacts/p4a_results.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
