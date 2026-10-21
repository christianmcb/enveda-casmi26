# Condensed v0.1 pipeline (from the Kaggle notebook "Version 0.1 - Baseline").
# NOT directly runnable: placeholders marked "..." must be filled from the notebook.
# Use as a reference when building modules in src/.

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit

RDLogger.DisableLog("rdApp.*")
rng = np.random.default_rng(42)
SMILES_COL, PROV_KEY = "normalized_smiles", "inchikey14"

# ---- clean ---------------------------------------------------------------
# ADDUCT = {adduct: (n_molecules, shift)}; shifts:
# [M+H]+ 1.007276, [M-H]- -1.007276, [M+Na]+ 22.989218, [M+NH4]+ 18.033823, [M+CH2O2-H]- 44.997655
df = df[df.adduct.isin(ADDUCT) & df.precursor_mz.between(50, 1500)
        & df.precursor_error_ppm.abs().le(20) & df.num_peaks.between(3, 500)].copy()
df["neutral_mass"] = (df.precursor_mz - df.adduct.map(lambda a: ADDUCT[a][1])) / df.adduct.map(lambda a: ADDUCT[a][0])
df["ce_multi"] = df.collision_energy_ev.apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).astype(np.float32)
df["collision_energy_ev"] = df.collision_energy_ev.apply(ce_scalar)  # mean of arrays, NaN on failure

# ---- subsample + canonical key + stereo-free scaffold + split ---------------
taut = rdMolStandardize.TautomerEnumerator()

def key_and_scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    mc = taut.Canonicalize(m)
    key = Chem.MolToInchiKey(mc)[:14]
    ms = Chem.Mol(mc)
    Chem.RemoveStereochemistry(ms)
    return key, MurckoScaffold.MurckoScaffoldSmiles(mol=ms)

# <=5,000 molecules per source, <=3 spectra per molecule (re-applied after canonicalisation)
# sub["key"], sub["scaffold"]; groupby("key").head(3); scaffold = mode per key
# acyclic: sub.loc[empty, "scaffold"] = "acyclic_" + key
# GroupShuffleSplit(test_size=0.2, random_state=42) on scaffold
# va = va[~va.key.isin(set(tr.key))]; assert no overlap
# CACHE the canonicalisation (took ~21 min for 30k SMILES).

# ---- features --------------------------------------------------------------
MAX_MZ, BIN = 1000, 1.0
NB = int(MAX_MZ / BIN)

def spec_vec(mz, it):
    v = np.zeros(NB, np.float32)
    idx = (mz / BIN).astype(int)
    ok = (idx >= 0) & (idx < NB)
    np.add.at(v, idx[ok], np.sqrt(it[ok]))
    return v / (np.linalg.norm(v) + 1e-9)

# X = [spec_vec | neutral_mass/1000, CE/100, ce_multi | one-hot ionization_mode,
#      instrument_type (top 8 + other), adduct]   (columns fixed from train only)

# ---- fingerprints, library ---------------------------------------------------
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
# mols_u = one row per canonical key; FP, mass_u (ExactMolWt); Ytr = FP[row_of[tr.key]]
# lib = train molecules only; all_* = train+val molecules (oracle library)
unit = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)

# ---- metric ----------------------------------------------------------------
def mrr_at_25(ranked, true):
    return float(np.mean([1 / (c[:25].index(t) + 1) if t in c[:25] else 0.0
                          for c, t in zip(ranked, true)]))

def top_k(S, k=25):
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return np.take_along_axis(idx, np.argsort(-np.take_along_axis(S, idx, 1), axis=1), 1)

# ---- model + ranking -------------------------------------------------------
reg = Ridge(alpha=1.0).fit(Xtr, Ytr)
P = unit(reg.predict(Xv))  # Xv = 2000 val spectra (seed this next time!)

# ---- benchmark: score only candidates inside the mass window ----------------
def eval_pool(keys, fp_n, mass, tol_ppm=10):
    order = np.argsort(mass)
    m_sorted = mass[order]
    sm, sr, nc = [], [], []
    for i in range(len(true_v)):
        q = mass_v[i]
        w = q * tol_ppm * 1e-6
        lo, hi = np.searchsorted(m_sorted, [q - w, q + w])
        idx = order[lo:hi]
        nc.append(len(idx))
        if len(idx) == 0:
            sm.append(0.0); sr.append(0.0); continue
        s = fp_n[idx] @ P[i]
        sm.append(rr([keys[j] for j in idx[np.argsort(-s)[:25]]], true_v[i]))
        sr.append(rr([keys[j] for j in rng.permutation(idx)[:25]], true_v[i]))
    return np.mean(nc), np.mean(sm), np.mean(sr)

# decoys: 200k molecules not in the subsample, keys prefixed "d_" so they can never equal a true key
