"""Rebuild the v0.1 subsample and cache the canonical key + stereo-free scaffold.

Why: v0.1 spent ~21 min per run on tautomer canonicalisation and its cache lived only on Kaggle.
This script reproduces the v0.1 selection (same filters, same seeds) from the small metadata
columns, canonicalises the unique SMILES in parallel, and writes

    artifacts/sub_rows_v1.parquet     row position in train.parquet + provisional columns of the v0.1 subsample
    artifacts/canon_cache_v1.parquet  smiles, key (InChIKey14 of canonical tautomer), scaffold (stereo removed)

Run from the repo root:  python src/build_canon_cache.py
Sanity target (docs/CONTEXT.md): 76,248 spectra / 29,964 molecules after canonicalisation.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

TRAIN_PATH = "data/train.parquet"
OUT = "artifacts"
SMILES_COL, PROV_KEY, SRC = "normalized_smiles", "inchikey14", "ingest_lib"
PER_SRC, PER_MOL = 5_000, 3
ADDUCTS = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+CH2O2-H]-"]   # v0.1 table with KEEP_DIMERS=False


def select_subsample():
    cols = [SRC, SMILES_COL, PROV_KEY, "adduct", "precursor_mz", "precursor_error_ppm", "num_peaks"]
    df = pq.read_table(TRAIN_PATH, columns=cols).to_pandas()      # index == row position in the file
    df = df[df.adduct.isin(ADDUCTS) & df.precursor_mz.between(50, 1500)
            & df.precursor_error_ppm.abs().le(20) & df.num_peaks.between(3, 500)].copy()
    print(f"cleaned: {len(df):,} spectra, {df[PROV_KEY].nunique():,} molecules")

    mol_tab = df.groupby(PROV_KEY, as_index=False)[SRC].first()
    keys = np.unique(np.concatenate([
        g[PROV_KEY].sample(min(len(g), PER_SRC), random_state=42).values
        for _, g in mol_tab.groupby(SRC)]))
    sub = (df[df[PROV_KEY].isin(keys)].rename_axis("row").reset_index()
           .sample(frac=1, random_state=0).groupby(PROV_KEY).head(PER_MOL).reset_index(drop=True))
    print(f"subsample: {len(sub):,} spectra | {sub[PROV_KEY].nunique():,} molecules (before canonicalisation)")
    return sub


def key_and_scaffold(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    global _taut
    try:
        _taut
    except NameError:
        _taut = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None, None
        mc = _taut.Canonicalize(m)
        key = Chem.MolToInchiKey(mc)[:14]
        ms = Chem.Mol(mc)
        Chem.RemoveStereochemistry(ms)                 # scaffold must ignore stereo (leak fix, CLAUDE.md)
        return key, MurckoScaffold.MurckoScaffoldSmiles(mol=ms)
    except Exception:
        return None, None


if __name__ == "__main__":
    import rdkit
    print("RDKit", rdkit.__version__, "(competition pins 2026.03.3)")
    os.makedirs(OUT, exist_ok=True)
    sub = select_subsample()
    sub.to_parquet(f"{OUT}/sub_rows_v1.parquet")

    smiles = sub[SMILES_COL].unique().tolist()
    with Pool(os.cpu_count()) as pool:
        res = list(tqdm(pool.imap(key_and_scaffold, smiles, chunksize=100), total=len(smiles)))
    cache = pd.DataFrame({"smiles": smiles, "key": [r[0] for r in res], "scaffold": [r[1] for r in res]})
    print("failed to parse:", cache.key.isna().sum(), "of", len(cache))
    cache.to_parquet(f"{OUT}/canon_cache_v1.parquet")

    s = sub.merge(cache, left_on=SMILES_COL, right_on="smiles").dropna(subset=["key", "scaffold"])
    s = s.groupby("key").head(PER_MOL)
    print(f"after canonicalisation: {len(s):,} spectra | {s.key.nunique():,} molecules  (v0.1: 76,248 / 29,964)")
