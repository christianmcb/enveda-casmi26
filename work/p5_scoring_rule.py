"""P5 diagnostic: 2x2 of {Ridge v0.2c, official MLP} x {cosine, Bernoulli log-likelihood} ranking. No training.
Declared before looking at results: loglik with p clipped to [1e-4, 1-1e-4]; no tuning.
Run from the repo root:  python work/p5_scoring_rule.py  -> artifacts/p5_rule_scores.csv, p5_rule_bootstrap.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

official = pd.read_csv("artifacts/p5_selection.csv").config.iloc[0]
D = v1.load_data(); B = bm.Bench(D)
P = {"ridge": np.load("artifacts/p4c_pred_v0.2c.npy"), "mlp": np.load(f"artifacts/p5_pred_{official}.npy")}
sc = []
for m, pred in P.items():
    for rule in ("cosine", "loglik"):
        sc.append(B.score(f"{m}+{rule}", pred, rule=rule))
        print(sc[-1][["version", "slice", "model", "model_minus_random", "fp_cosine"]].round(4).to_string(index=False), flush=True)
sc = pd.concat(sc); sc.to_csv("artifacts/p5_rule_scores.csv", index=False)
boot = []
for sl in ("enveda180", "mixed_2000"):
    for a, b in [("ridge+cosine", "ridge+loglik"), ("mlp+cosine", "mlp+loglik"), ("ridge+cosine", "mlp+loglik"), ("ridge+loglik", "mlp+loglik")]:
        d, (lo, hi) = B.paired_diff(a, b, sl); boot.append(dict(slice=sl, baseline=a, variant=b, diff=d, ci_lo=lo, ci_hi=hi))
boot = pd.DataFrame(boot); boot.to_csv("artifacts/p5_rule_bootstrap.csv", index=False)
pd.set_option("display.width", 250)
print(sc.pivot_table(index="version", columns="slice", values="model", sort=False).round(4)); print(boot.round(4).to_string(index=False))
