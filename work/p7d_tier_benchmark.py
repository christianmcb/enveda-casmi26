"""P7d benchmark: the PubChem mass-window channel on labelled validation molecules (deep MLP tier-D predictions, molecule-level pooling).

Question: ~70% of hidden-test molecules are in neither train nor COCONUT. Can PubChem rescue them, and what does the channel cost for the molecules we already reach?
For each sampled validation molecule (truth known) we rank, with the notebook's own code (cell `tier_defs` of work/kaggle_cells.py):
  A  truth IN the library, tier OFF            current behaviour (reference)
  B  truth REMOVED from the library, tier ON   molecules reachable only through PubChem (the gain scenario)
  C  truth IN the library, tier ON             molecules we already reach (the cost scenario)
for several popularity weights w. Library = the project's 230k benchmark pool (the truth is a candidate in A and C, removed in B).
Slices: nonE = non-enveda-180 sources (natural-product-like chemistry, 10 ppm window; the relevant slice), E180 = enveda-180 (P3 window).
Caveats printed with the results: val truths are well-documented known compounds, so the prior helps here more than for never-isolated test molecules.
Run from the repo root (RDKit 2026.03.3):  python work/p7d_tier_benchmark.py [n_nonE n_E180]   -> artifacts/p7d_tier_benchmark.csv
"""
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, "src"); sys.path.insert(0, "work")
import benchmark as bm
import kaggle_cells
import mass_window as mw
import v1_pipeline as v1

N_NONE, N_E180 = (int(x) for x in sys.argv[1:3]) if len(sys.argv) >= 3 else (500, 250)
WEIGHTS = [0.15, 0.3, 0.5]
BONUSES = [0.0, 2.0, 4.0, 6.0]                      # source prior for library candidates (z-score units)
T0 = time.time()
def tick(m): print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)

g = globals()                     # the notebook cells run in __main__ so that their functions can be pickled by multiprocessing
g.update({"WORK": ".", "DEBUG": False})
for name in ("helpers_mass", "ranking", "tier_defs"):                    # exactly the notebook's code
    exec(next(src for k, n, src in kaggle_cells.CELLS if n == name), g)

tier = g["Tier"](g["find_tier_dir"]())
tick(f"tier: {len(tier.mass):,} structures, mass {tier.mass[0]:.2f}-{tier.mass[-1]:.2f}")

D = v1.load_data(); B = bm.Bench(D); va = D["va"]
P = v1.unit(np.load("artifacts/scale/pred_deep_D.npy"))
pop_tab = pd.read_csv("data/external/pop_prior/pool_popularity.csv", usecols=["inchikey14", "pop"]).drop_duplicates("inchikey14").set_index("inchikey14")["pop"]
lib = g["Library"](list(B.keys), B.keys, B.mass, B.fp)                         # smiles = keys here (the benchmark pool stores no SMILES)
pop_lib = pop_tab.reindex(lib.keys).values.astype(np.float32); pop_lib[np.isnan(pop_lib)] = np.nanmedian(pop_lib); lib.pop = pop_lib
raw_smiles = pd.Series(D["mols_u"][v1.SMILES_COL].values, index=D["mols_u"].key.values)

def raw_ik14(smi):
    m = Chem.MolFromSmiles(smi); return Chem.MolToInchiKey(m)[:14] if m is not None else None

def build_queries(mask, n, tol, bias, seed=42):
    """Pooled molecule-level queries for a seeded sample of validation molecules of one slice."""
    q = va[mask]; keys = np.array(sorted(q.key.unique())); rng = np.random.default_rng(seed)
    keys = rng.choice(keys, min(n, len(keys)), replace=False)
    idx_of = {k: np.flatnonzero(mask & (va.key.values == k)) for k in keys}
    Pm, lo, hi = [], [], []
    for k in keys:
        ix = idx_of[k]; d = va.iloc[ix]
        l, h = mw.window(d.neutral_mass.values, d.adduct.values, tol, bias)
        L, H = l.max(), h.min()
        if L > H: L, H = l.min(), h.max()
        Pm.append(P[ix].mean(0)); lo.append(L); hi.append(H)
    return keys, v1.unit(np.stack(Pm)), np.array(lo), np.array(hi)

def mrr(ranked, truth):
    rr = []
    for r, t in zip(ranked, truth):
        ks = [k for k, _, _ in r]; rr.append(1 / (ks.index(t) + 1) if t in ks else 0.0)
    return np.array(rr)

nonE = (va.ingest_lib != "enveda-180").values; e180 = (va.ingest_lib == "enveda-180").values
slices = {"nonE": build_queries(nonE, N_NONE, 10.0, None), "E180": build_queries(e180, N_E180, B.HW, B.BIAS)}
tick("queries: " + ", ".join(f"{s} {len(v[0])}" for s, v in slices.items()))

pool = mp.get_context("fork").Pool(os.cpu_count())
fp_cache, key_cache, libset = {}, {}, set(lib.keys)
rows = []
for sname, (keys, Pm, lo, hi) in slices.items():
    # diagnostic: does the raw InChIKey14 of the truth occur in the tier window at all?
    raw = [raw_ik14(raw_smiles[k]) for k in keys]
    inwin = np.array([(r in set(tier.ik14(np.arange(*tier.window(l, h))))) for r, l, h in zip(raw, lo, hi)])
    nwin = np.array([tier.window(l, h)[1] - tier.window(l, h)[0] for l, h in zip(lo, hi)])
    tick(f"[{sname}] {len(keys)} molecules | truth's raw key found in the tier window: {inwin.mean():.1%} | tier candidates per window: median {np.median(nwin):.0f}, mean {nwin.mean():.0f}, max {nwin.max()}")
    for w in WEIGHTS:
        lib.pop_w = w
        ra, _ = g["rank_queries"](Pm, lo, hi, lib)
        rr_a = mrr([[(lib.keys[j], None, "lib") for j in r] for r in ra], keys)
        rows.append(dict(slice=sname, scenario="A", w=w, bonus=np.nan, mrr=rr_a.mean(), n=len(keys), truth_in_tier_window=inwin.mean()))
        for bonus in BONUSES:
            res = {}
            for scen, excl in (("B", True), ("C", False)):               # B: truth NOT in library (PubChem-only), C: truth in library and tier on
                ranked = []
                for i, k in enumerate(keys):
                    if excl:                                                    # remove the truth from the library for this query only
                        sub = g["Library"](lib.smiles, lib.keys, np.where(lib.keys != k, lib.mass, -1.0), lib.fp_unit); sub.pop = lib.pop; sub.pop_w = w; used = sub
                    else:
                        used = lib
                    ranked += g["rank_with_tier"](Pm[i:i + 1], lo[i:i + 1], hi[i:i + 1], used, tier, w, fp_cache=fp_cache, key_cache=key_cache, pool=pool,
                                                  lib_keyset=(libset - {k}) if excl else libset, lib_bonus=bonus)
                rr = mrr(ranked, keys); res[scen] = rr.mean()
                rows.append(dict(slice=sname, scenario=scen, w=w, bonus=bonus, mrr=rr.mean(), n=len(keys), truth_in_tier_window=inwin.mean(),
                                 tier_guess_share=np.mean([src == "tier" for r in ranked for _, _, src in r])))
            tick(f"[{sname}] w={w} bonus={bonus}: A {rr_a.mean():.4f} | B {res['B']:.4f} | C {res['C']:.4f}")
            pd.DataFrame(rows).to_csv("artifacts/p7d_tier_benchmark.csv", index=False)
pool.close()
print(pd.DataFrame(rows).round(4).to_string(index=False))
