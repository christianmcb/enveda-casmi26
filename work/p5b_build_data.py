"""P5b: build LARGER training sets for the learning-curve test, without touching the validation set.

Tiers (nested; validation rows/molecules are identical to every earlier experiment):
  A25 < A50 < A (current 61,701 spectra) < B (per-source molecule cap 15,000) < D (ALL cleaned molecules, <=3 spectra each)
New training molecules are canonicalised (same recipe as the metric) and DROPPED if their canonical key OR stereo-free Murcko
scaffold occurs in validation (scaffold-disjoint split preserved), or if their key is already in the current training set.

Run from the repo root:  python work/p5b_build_data.py [--dry]      (--dry: tiny extras, ~5 min, writes to artifacts/scale_dry)
-> artifacts/scale/{X_train.npz, X_val.npz, rows_train.parquet, mol_fp.npy, canon_extra.parquet}
"""
import os
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

sys.path.insert(0, "src")
sys.path.insert(0, "work")
import v1_pipeline as v1
from build_canon_cache import key_and_scaffold

DRY = "--dry" in sys.argv
OUT = "artifacts/scale_dry" if DRY else "artifacts/scale"
os.makedirs(OUT, exist_ok=True)
SMILES, PROV, SRC = v1.SMILES_COL, "inchikey14", "ingest_lib"
CAP_B = 15_000

# ---- 1. current split (A = current train, va = validation) -----------------------------------------------------------
D = v1.load_data()
tr, va = D["tr"], D["va"]
val_keys, val_scaf, tr_keys = set(va.key), set(va.scaffold), set(tr.key)
sub_prov = set(D["sub"][PROV])

# ---- 2. candidate extra rows from the cleaned data ----------------------------------------------------------------------
cols = [SRC, SMILES, PROV, "adduct", "precursor_mz", "precursor_error_ppm", "num_peaks"]
df = pq.read_table(v1.TRAIN_PATH, columns=cols).to_pandas()
df = df[df.adduct.isin(v1.ADDUCT) & df.precursor_mz.between(50, 1500) & df.precursor_error_ppm.abs().le(20)
        & df.num_peaks.between(3, 500)].rename_axis("row").reset_index()
unused = df[~df[PROV].isin(sub_prov)]
mol_src = unused.groupby(PROV)[SRC].first()
rng = np.random.default_rng(42)
mols = mol_src.index.values
if DRY:
    mols = rng.choice(mols, 4000, replace=False)
a_counts = D["sub"].drop_duplicates(PROV).groupby(SRC).size()
inB = set()
for s_name, g in mol_src.loc[mols].groupby(mol_src.loc[mols]):
    need = max(0, CAP_B - int(a_counts.get(s_name, 0))) if not DRY else 800
    inB |= set(rng.choice(g.index.values, min(need, len(g)), replace=False))
extra = (unused[unused[PROV].isin(mols)].sample(frac=1, random_state=0).groupby(PROV).head(3).reset_index(drop=True))
print(f"extra candidate rows: {len(extra):,} spectra | {extra[PROV].nunique():,} molecules ({len(inB):,} of them in tier B)", flush=True)

# ---- 3. canonicalise (cached, parallel) -----------------------------------------------------------------------------------
cpath = f"{OUT}/canon_extra.parquet"
cache = pd.read_parquet(cpath) if os.path.exists(cpath) else pd.DataFrame(columns=["smiles", "key", "scaffold"])
todo = [s for s in extra[SMILES].unique() if s not in set(cache.smiles)]
print(f"canonicalising {len(todo):,} new SMILES ({len(cache):,} cached)", flush=True)
if todo:
    with get_context("fork").Pool(os.cpu_count()) as pool:     # fork: workers must not re-import this script
        res = list(tqdm(pool.imap(key_and_scaffold, todo, chunksize=200), total=len(todo), mininterval=30))
    cache = pd.concat([cache, pd.DataFrame({"smiles": todo, "key": [r[0] for r in res], "scaffold": [r[1] for r in res]})])
    cache.to_parquet(cpath)
ex = extra.merge(cache, left_on=SMILES, right_on="smiles", how="left").dropna(subset=["key", "scaffold"]).reset_index(drop=True)
ex["scaffold"] = ex.groupby("key").scaffold.transform(lambda s: s.mode().iloc[0])
empty = ex.scaffold == ""
ex.loc[empty, "scaffold"] = "acyclic_" + ex.loc[empty, "key"]

# ---- 4. leakage guards -----------------------------------------------------------------------------------------------------
n0 = ex.key.nunique()
ex = ex[~ex.key.isin(val_keys) & ~ex.scaffold.isin(val_scaf) & ~ex.key.isin(tr_keys)].reset_index(drop=True)
ex = ex.groupby("key").head(3).reset_index(drop=True)
assert not (set(ex.key) & val_keys) and not (set(ex.scaffold) & val_scaf), "validation leaked into the extra training set"
print(f"after leakage guards: {len(ex):,} spectra | {ex.key.nunique():,} molecules (dropped {n0 - ex.key.nunique():,} molecules: "
      f"val scaffold/key overlap, duplicates of current train)", flush=True)
ex["inB"] = ex[PROV].isin(inB)

# ---- 5. molecule fingerprint table (Morgan r2, 1024 bits, as uint8) --------------------------------------------------------
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
first = pd.concat([D["mols_u"][["key", SMILES]], ex.drop_duplicates("key")[["key", SMILES]]]).drop_duplicates("key").reset_index(drop=True)
fps = np.zeros((len(first), 1024), np.uint8); ok = np.ones(len(first), bool)
for i, s_ in enumerate(tqdm(first[SMILES], mininterval=30)):
    m = Chem.MolFromSmiles(s_)
    if m is None: ok[i] = False
    else: fps[i] = fpgen.GetFingerprintAsNumPy(m)
first, fps = first[ok].reset_index(drop=True), fps[ok]
mol_idx = pd.Series(np.arange(len(first)), index=first.key)
ex = ex[ex.key.isin(mol_idx.index)].reset_index(drop=True)
np.save(f"{OUT}/mol_fp.npy", fps)
print(f"fingerprint table: {len(first):,} molecules", flush=True)

# ---- 6. features in the full 32,019-column space ------------------------------------------------------------------------------
feat = v1.Features(tr, neutral_loss=True, frag_defect=0.02, loss_defect=0.02)
X_A = feat.build(tr, D["mz_t"], D["it_t"]).tocsr()
X_val = feat.build(va, D["mz_v"], D["it_v"]).tocsr()
del D["mz_t"], D["it_t"], D["mz_v"], D["it_v"]

def stream(rows):
    """One pass over the parquet row groups: build features for `rows` (a frame with file position `row`), never holding all peaks."""
    pf = pq.ParquetFile(v1.TRAIN_PATH)
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    rid = rows["row"].values
    parts, pos = [], []
    for g in range(pf.num_row_groups):
        m = np.flatnonzero((rid >= starts[g]) & (rid < starts[g + 1]))
        if len(m) == 0: continue
        t = (pf.read_row_group(g, columns=["ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev", "instrument_type", "ionization_mode"])
             .take(rid[m] - starts[g]).to_pandas())
        d = rows.iloc[m][["adduct", "precursor_mz"]].reset_index(drop=True).copy()
        d["instrument_type"], d["ionization_mode"] = t.instrument_type.values, t.ionization_mode.values
        ce = [v1._ce(x) for x in t.collision_energy_ev]
        d["collision_energy_ev"], d["ce_multi"] = [c[0] for c in ce], np.float32([c[1] for c in ce])
        d["neutral_mass"] = (d.precursor_mz - d.adduct.map(lambda a: v1.ADDUCT[a][1])) / d.adduct.map(lambda a: v1.ADDUCT[a][0])
        parts.append(feat.build(d, t.ms2_mzs.values, t.ms2_normalized_intensities.values).tocsr()); pos.append(m)
        print(f"  row group {g + 1}/{pf.num_row_groups}: {len(m):,} rows", flush=True)
    X = sp.vstack(parts).tocsr(); order = np.argsort(np.concatenate(pos))
    return X[order]
X_ex = stream(ex)
X_train = sp.vstack([X_A, X_ex]).tocsr()
sp.save_npz(f"{OUT}/X_train.npz", X_train); sp.save_npz(f"{OUT}/X_val.npz", X_val)

# ---- 7. row table with nested tier flags ---------------------------------------------------------------------------------------
a_rows = tr[["key", "scaffold", SRC]].assign(is_A=True, inB=False)
e_rows = ex[["key", "scaffold", SRC, "inB"]].assign(is_A=False)
R = pd.concat([a_rows, e_rows]).reset_index(drop=True)
ukeys = np.random.default_rng(42).permutation(tr.key.unique())
k25, k50 = set(ukeys[: len(ukeys) // 4]), set(ukeys[: len(ukeys) // 2])
R["in_A25"] = R.is_A & R.key.isin(k25); R["in_A50"] = R.is_A & R.key.isin(k50)
R["in_A"] = R.is_A; R["in_B"] = R.is_A | R.inB; R["in_D"] = True
R["mol_idx"] = mol_idx[R.key].values
R.to_parquet(f"{OUT}/rows_train.parquet")
val_idx = mol_idx[va.key].values; np.save(f"{OUT}/val_mol_idx.npy", val_idx)
print(R[["in_A25", "in_A50", "in_A", "in_B", "in_D"]].sum().to_string(), "\nunique molecules per tier:",
      {t: int(R[R[f"in_{t}"]].key.nunique()) for t in ["A25", "A50", "A", "B", "D"]}, "\nX_train", X_train.shape, "X_val", X_val.shape, flush=True)
