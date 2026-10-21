"""P4c: mass-defect-aware sub-bins (nominal mass x 0.02 Da defect sub-bins inside [-0.05, +0.25)) vs v0.2b (uniform 0.1 Da).
Single change: the binning scheme of both blocks. Everything else (Ridge alpha 1, pool, windows, eval) identical.

Run from the repo root:  python work/p4c_defect_bins.py   -> artifacts/p4c_results.csv, artifacts/p4c_bootstrap.csv
Needs ~10 GB RAM. Models are fitted, scored and freed one at a time (their Gram matrices are several GB each).
"""
import gc
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

# ---- unit checks -------------------------------------------------------------------------------------------
col = lambda x: int(v1.defect_block([np.array([x])], [np.ones(1)], w=0.02).indices[0])
print("columns per nominal mass:", v1.defect_block([np.array([100.0])], [np.ones(1)]).shape[1] // 1000)
assert col(108.9985) == col(109.0012), "peaks straddling an integer must share a sub-bin"
assert col(109.0012) // 16 == 109 and col(108.9985) // 16 == 109, "nominal mass is floor(x + 0.05)"
assert col(109.2) % 16 == 12 and col(109.31) % 16 == 15 and col(109.9) % 16 == 15, "defect >= 0.25 goes to the overflow column"
assert col(109.00) != col(109.03), "0.02 Da sub-bins separate a 30 mDa difference"
print("defect_block unit checks passed", flush=True)

D = v1.load_data()
B = bm.Bench(D)
res = []

def run(name, alphas=(1.0,), **kw):
    M = v1.fit_ridge(D, alpha=alphas[0], neutral_loss=True, solver="normal_eq", **kw)
    print(f"{name}: {M['Xtr'].shape[1]:,} columns", flush=True)
    for a in alphas:
        if a != alphas[0]:
            try:
                M["reg"].refit(a)
            except RuntimeError as e:
                print("   skipped alpha", a, "-", e); continue
        label = name if a == alphas[0] and len(alphas) == 1 or a == alphas[0] else f"{name}_alpha{a:g}"
        res.append(B.score(label, M["reg"].predict(M["Xva"])).assign(alpha=a, columns=M["Xtr"].shape[1]))
        print(res[-1][["version", "slice", "model", "model_minus_random", "fp_cosine"]].round(4).to_string(index=False), flush=True)
        pd.concat(res).to_csv("artifacts/p4c_results.csv", index=False)
    del M; gc.collect()

run("v0.2b", frag_bin=0.1, loss_bin=0.1)                                   # reference; reproduces 0.3214 / 0.4839 with the new solver
run("v0.2c_defect0.02", alphas=(1.0, 3.0, 10.0), frag_defect=0.02, loss_defect=0.02)   # OFFICIAL P4c
run("diag_defect0.05_both", frag_defect=0.05, loss_defect=0.05)
run("diag_defect0.02_frag_only", frag_defect=0.02, loss_bin=0.1)           # losses stay at uniform 0.1 Da

out = pd.concat(res); out.to_csv("artifacts/p4c_results.csv", index=False)
boot = []
for sname in ("enveda180", "mixed_2000"):
    for v in [r for r in out.version.unique() if r != "v0.2b"]:
        d, (lo, hi) = B.paired_diff("v0.2b", v, sname)
        boot.append(dict(slice=sname, baseline="v0.2b", variant=v, diff=d, ci_lo=lo, ci_hi=hi))
pd.DataFrame(boot).to_csv("artifacts/p4c_bootstrap.csv", index=False)
pd.set_option("display.width", 250)
print(out.pivot_table(index="version", columns="slice", values="model", sort=False).round(4))
print(pd.DataFrame(boot).round(4).to_string(index=False))
