"""Plumbing check for the FPNet cell (work/kaggle_cells.py, cell `fpnet_defs`) -- NOT a score estimate.
The visible test.parquet is a sample of train (every spectrum has an exact twin), so the truth structure of each molecule_id can be recovered from the
twin rows; the public FPNet (fpnet_full1) was trained on those molecules, so the MRR here is leaked and only shows that queries, bit layout, key join,
windows and ranking are wired correctly (expect far above the 0.15 leaderboard level; a near-zero value means a bug).
Library = FPNet's own 710,701-structure pool (so every truth is present). Run from the repo root:
  python work/test_fpnet_plumbing.py [n_molecules]     (needs PyTorch and RDKit 2026.3.3)"""
import glob, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "work")
import kaggle_cells
from build_canon_cache import key_and_scaffold

N_MOL = int(sys.argv[1]) if len(sys.argv) > 1 else 400
t0 = time.time()
tick = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {m}", flush=True)
test = pd.read_parquet("data/test.parquet")
keep_ids = test.molecule_id.drop_duplicates().iloc[:N_MOL]; test = test[test.molecule_id.isin(keep_ids)].reset_index(drop=True)

# ---- truth from the twin rows in train ----
tr = pd.read_parquet("data/train.parquet", columns=["precursor_mz", "num_peaks", "adduct", "normalized_smiles"])
tr["k"] = list(zip(tr.precursor_mz.round(4), tr.num_peaks, tr.adduct))
want = set(zip(test.precursor_mz.round(4), test.ms2_mzs.map(len), test.adduct))
tr = tr[tr.k.isin(want)]
smi_of = tr.groupby("k").normalized_smiles.agg(lambda s: s.value_counts().index[0])
test["twin_smiles"] = [smi_of.get((round(p, 4), len(m), a)) for p, m, a in zip(test.precursor_mz, test.ms2_mzs, test.adduct)]
mol_smiles = test.dropna(subset=["twin_smiles"]).groupby("molecule_id").twin_smiles.agg(lambda s: s.value_counts().index[0])
truth = {m: key_and_scaffold(s)[0] for m, s in mol_smiles.items()}
tick(f"truth recovered for {len(truth)}/{test.molecule_id.nunique()} molecules")

# ---- notebook cells, executed exactly as generated ----
import torch
g = dict(np=np, pd=pd, os=os, glob=glob, sys=sys, torch=torch, tick=tick, test=test, FPNET_MODELS=os.environ.get("CASMI_FPNET_MODELS", "full1"))
for name in ("helpers_mass", "ranking", "fpnet_defs"):
    exec(next(src for k, n, src in kaggle_cells.CELLS if n == name), g)
pol = pd.read_csv("artifacts/mass_policy_v1.csv").set_index("adduct")
BIAS, HW = pol.bias_ppm.to_dict(), pol.halfwidth_ppm.to_dict()
meta = pd.read_parquet(glob.glob("data/external/fpnet/**/pool_meta.parquet", recursive=True)[0])
meta = meta.drop_duplicates("key").reset_index(drop=True); meta = meta.iloc[np.argsort(meta.mass.values, kind="stable")] if False else meta
LIB = g["Library"](meta.smiles.tolist(), meta.key.values, meta.mass.values, np.zeros((len(meta), 2), np.float32))
pp = pd.read_csv("data/external/pop_prior/pool_popularity.csv", usecols=["inchikey14", "pop"]).drop_duplicates("inchikey14").set_index("inchikey14")["pop"]
pop = pp.reindex(LIB.keys).values.astype(np.float32); pop[np.isnan(pop)] = np.nanmedian(pop)
nm = test.precursor_mz.values - test.adduct.map(g["SHIFT"]).values
test_lo, test_hi = g["mass_window"](nm, test.adduct.values, HW, BIAS)
sample = pd.DataFrame({"molecule_id": test.molecule_id.drop_duplicates().values})
g.update(LIB=LIB, test=test, test_lo=test_lo, test_hi=test_hi, sample=sample)
FPS = g["FPNetScorer"](LIB.keys, "cpu")
keys, Pm, L, H, nbad = g["pool_molecules"](np.zeros((len(test), 2), np.float32) + 1.0, test.molecule_id.values, test_lo, test_hi)
Z = FPS.molecule_logits(test)

def mrr(ranked):
    rr = [next((1 / (r + 1) for r, j in enumerate(rk) if LIB.keys[j] == truth[m]), 0.0) for m, rk in zip(keys, ranked) if m in truth]
    return float(np.mean(rr)), len(rr)
for pw in (0.0, 0.15):
    LIB.pop, LIB.pop_w = (pop, pw) if pw else (None, 0.0)
    ranked, ncand, nofp = g["rank_queries_fpnet"](keys, Pm, L, H, LIB, FPS, Z, 0.0)
    m, n = mrr(ranked)
    print(f"FPNet only, pop weight {pw}: MRR@25 {m:.4f} on {n} molecules (LEAKED, plumbing only) | median candidates {np.median(ncand):.0f} | rows without FPNet fp {nofp}", flush=True)

# ---- the notebook's own submission writer (validate_submission, diagnostics) ----
g["WORK"] = "kaggle_work_debug"; g["DEBUG"] = True
LIB.pop, LIB.pop_w = pop, 0.15
g["LIB"] = LIB
sub, ranked, _ = g["make_submission_fpnet"](np.ones((len(test), 2), np.float32), FPS, 0.5, name="fpnet_plumbing_w0.5")
print(sub.head(3).assign(smiles=sub.smiles.str.split(";").str[:2].str.join(";")).to_string(index=False))
