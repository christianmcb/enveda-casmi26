"""P4b: finer m/z bins (0.1 Da, sparse) on top of v0.2a (fragment + neutral-loss blocks). Single change: the bin width.

Run from the repo root:  python work/p4b_fine_bins.py   -> artifacts/p4b_results.csv, artifacts/p4b_bootstrap.csv
Diagnostics (not versions): 0.5 Da, and fine bins on only one block, plus a Ridge-alpha sweep for the official config.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

D = v1.load_data()
B = bm.Bench(D)

# ---- unit checks: the new code paths must reproduce the validated ones --------------------------------------
sel = np.arange(0, len(D["va"]), 70)[:200]
mz, it, pm = D["mz_v"][sel], D["it_v"][sel], D["va"].precursor_mz.values[sel]
dense_f = np.stack([v1.spec_vec(m, i) for m, i in zip(mz, it)])
dense_l = np.stack([v1.loss_vec(m, i, p) for m, i, p in zip(mz, it, pm)])
sp_f = v1.sparse_block(list(mz), it, 1.0, 1000, eps=0).toarray()
sp_l = v1.sparse_block([np.where(p - np.asarray(m) >= 1.0, p - np.asarray(m), -1.0) for m, p in zip(mz, pm)], it, 1.0, 1000, eps=0).toarray()
sp_l_eps = v1.sparse_block([np.where(p - np.asarray(m) >= 1.0, p - np.asarray(m), -1.0) for m, p in zip(mz, pm)], it, 1.0, 1000).toarray()
print("with the float guard (eps=1e-6), loss block differs in", int((np.abs(dense_l - sp_l_eps).max(1) > 1e-5).sum()), "of", len(sel), "spectra (peaks sitting exactly on a bin edge)")
print("sparse_block == spec_vec/loss_vec at 1 Da (eps=0):", np.abs(dense_f - sp_f).max(), np.abs(dense_l - sp_l).max(), flush=True)
assert np.abs(dense_f - sp_f).max() < 1e-5 and np.abs(dense_l - sp_l).max() < 1e-5

cache, res = {}, []
def run(name, alpha=1.0, **kw):
    M = v1.fit_ridge(D, alpha=alpha, cache=cache, neutral_loss=True, **kw)
    res.append(B.score(name, M["reg"].predict(M["Xva"])).assign(alpha=alpha, **{k: kw.get(k) for k in ("frag_bin", "loss_bin", "solver")}))
    print(res[-1][["version", "slice", "model", "model_minus_random", "fp_cosine"]].round(4).to_string(index=False), flush=True)
    pd.concat(res).to_csv("artifacts/p4b_results.csv", index=False)
    return M

run("v0.2a", solver="sklearn")                                        # reference (must reproduce 0.2767 / 0.4469)
M_ne = run("v0.2a_normal_eq_check", solver="normal_eq", frag_bin=1.0, loss_bin=1.0)   # solver check on identical features
Ms = v1.fit_ridge(D, neutral_loss=True)
print("max |normal_eq - sklearn| prediction difference:", float(np.abs(M_ne["reg"].predict(M_ne["Xva"]) - Ms["reg"].predict(Ms["Xva"])).max()), flush=True)

run("v0.2b_bin0.1", solver="normal_eq", frag_bin=0.1, loss_bin=0.1)    # OFFICIAL P4b
run("diag_bin0.5", solver="normal_eq", frag_bin=0.5, loss_bin=0.5)
run("diag_frag0.1_loss1", solver="normal_eq", frag_bin=0.1, loss_bin=1.0)
run("diag_frag1_loss0.1", solver="normal_eq", frag_bin=1.0, loss_bin=0.1)
for a in (3.0, 10.0, 30.0, 100.0):                                    # does the fine-bin model just need more regularisation?
    run(f"diag_bin0.1_alpha{a:g}", alpha=a, solver="normal_eq", frag_bin=0.1, loss_bin=0.1)

out = pd.concat(res); out.to_csv("artifacts/p4b_results.csv", index=False)
boot = []
for sname in ("enveda180", "mixed_2000"):
    for v in [r for r in out.version.unique() if r not in ("v0.2a", "v0.2a_normal_eq_check")]:
        d, (lo, hi) = B.paired_diff("v0.2a", v, sname)
        boot.append(dict(slice=sname, baseline="v0.2a", variant=v, diff=d, ci_lo=lo, ci_hi=hi))
pd.DataFrame(boot).to_csv("artifacts/p4b_bootstrap.csv", index=False)
pd.set_option("display.width", 250)
print(out.pivot_table(index="version", columns="slice", values="model", sort=False).round(4))
print(pd.DataFrame(boot).round(4).to_string(index=False))
