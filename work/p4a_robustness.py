"""P4a robustness: is the neutral-loss gain real, and which block carries it?  (diagnostics, not new versions)

Run from the repo root:  python work/p4a_robustness.py   -> artifacts/p4a_robustness.csv, artifacts/p4a_bootstrap.csv
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

D = v1.load_data()
B = bm.Bench(D)
cache, res = {}, []
cfgs = [("frag", dict(), 1.0), ("frag+loss", dict(neutral_loss=True), 1.0), ("loss_only", dict(neutral_loss=True, fragment=False), 1.0)]
cfgs += [(f"{n}_a{a}", kw, a) for n, kw in [("frag", {}), ("frag+loss", dict(neutral_loss=True))] for a in (3.0, 10.0)]
for name, kw, alpha in cfgs:
    M = v1.fit_ridge(D, alpha=alpha, cache=cache, **kw)
    res.append(B.score(name if "_a" in name else f"{name}_a1.0", M["reg"].predict(M["Xva"])).assign(alpha=alpha))
    print(res[-1][["version", "slice", "model", "model_minus_random", "fp_cosine"]].round(4).to_string(index=False), flush=True)
out = pd.concat(res)
out.to_csv("artifacts/p4a_robustness.csv", index=False)

boot = []
for sname in ("enveda180", "mixed_2000"):
    for a, b in [("frag_a1.0", "frag+loss_a1.0"), ("frag_a1.0", "loss_only_a1.0"), ("frag_a3.0", "frag+loss_a3.0"), ("frag_a10.0", "frag+loss_a10.0")]:
        d, (lo, hi) = B.paired_diff(a, b, sname)
        boot.append(dict(slice=sname, baseline=a, variant=b, diff=d, ci_lo=lo, ci_hi=hi))
boot = pd.DataFrame(boot)
boot.to_csv("artifacts/p4a_bootstrap.csv", index=False)
pd.set_option("display.width", 250)
print(out.pivot_table(index="version", columns="slice", values="model").round(4))
print(boot.round(4).to_string(index=False))
