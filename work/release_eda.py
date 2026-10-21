"""Aggregate statistics of the training file for the release notebook (counts and histograms only: no spectra, no structures).
Run from the repo root (needs RDKit for formula masses):  python work/release_eda.py   -> artifacts/release_eda.json"""
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem

SHIFT = {"[M+H]+": 1.007276, "[M-H]-": -1.007276, "[M+Na]+": 22.989218, "[M+NH4]+": 18.033823, "[M+CH2O2-H]-": 44.997655}
PT = Chem.GetPeriodicTable(); _fm = {}
def formula_mass(f):
    if f in _fm: return _fm[f]
    try: m = sum(PT.GetMostCommonIsotopeMass(el) * (int(n) if n else 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f))
    except Exception: m = np.nan
    _fm[f] = m; return m

cols = ["ingest_lib", "inchikey14", "molecular_formula", "adduct", "precursor_mz", "num_peaks", "instrument_type", "ionization_mode"]
df = pq.read_table("data/train.parquet", columns=cols).to_pandas()
print(len(df), "spectra", flush=True)
out = {"n_spectra": int(len(df)), "n_molecules": int(df.inchikey14.nunique())}
g = df.groupby("ingest_lib").agg(spectra=("inchikey14", "size"), molecules=("inchikey14", "nunique")).sort_values("spectra", ascending=False)
out["by_source"] = g.reset_index().to_dict("list")
out["by_adduct"] = df.adduct.value_counts().head(8).to_dict()
out["by_instrument"] = df.instrument_type.value_counts().head(6).to_dict()
out["polarity"] = df.ionization_mode.value_counts().to_dict()
edges = np.unique(np.round(np.logspace(0, np.log10(80000), 60)))
for name, sel in (("all", np.ones(len(df), bool)), ("enveda-180", (df.ingest_lib == "enveda-180").values)):
    h, _ = np.histogram(df.num_peaks.values[sel], bins=edges); out[f"peaks_hist_{name}"] = h.tolist()
out["peaks_edges"] = edges.tolist()
out["peaks_median"] = float(df.num_peaks.median()); out["peaks_share_over_500"] = float((df.num_peaks > 500).mean())

# signed mass error from the formula (precursor_error_ppm is unsigned in the file) on enveda-180
e = df[(df.ingest_lib == "enveda-180") & df.adduct.isin(SHIFT)].copy()
e["M"] = e.molecular_formula.map(formula_mass); e = e.dropna(subset=["M"])
e["err"] = ((e.precursor_mz - e.adduct.map(SHIFT)) - e.M) / e.M * 1e6
bins = np.arange(-8, 8.01, 0.2)
out["ppm_edges"] = bins.tolist()
for ad in ("[M+H]+", "[M-H]-"):
    v = e.err[e.adduct == ad].values; h, _ = np.histogram(np.clip(v, -8, 8), bins=bins)
    out[f"ppm_hist_{ad}"] = h.tolist(); out[f"ppm_median_{ad}"] = float(np.median(v)); out[f"ppm_n_{ad}"] = int(len(v))
print("ppm done", flush=True)

# neutral mass of molecules by group (one value per molecule)
mol = df.drop_duplicates("inchikey14")[["inchikey14", "ingest_lib", "molecular_formula"]].copy()
mol["M"] = mol.molecular_formula.map(formula_mass); mol = mol.dropna(subset=["M"])
medges = np.arange(0, 1301, 25.0); out["mass_edges"] = medges.tolist()
for name, sel in (("enveda-180", mol.ingest_lib == "enveda-180"), ("other sources", mol.ingest_lib != "enveda-180")):
    h, _ = np.histogram(mol.M[sel].values, bins=medges); out[f"mass_hist_{name}"] = h.tolist()
out["mass_median_enveda180"] = float(mol.M[mol.ingest_lib == "enveda-180"].median()); out["mass_median_other"] = float(mol.M[mol.ingest_lib != "enveda-180"].median())
print("mass done", flush=True)

# 2-D histogram: fragment m/z vs mass defect (x - floor(x + 0.05)), intensity-weighted, from a sample of row groups
pf = pq.ParquetFile("data/train.parquet"); rng = np.random.default_rng(42)
groups = rng.choice(pf.num_row_groups, min(8, pf.num_row_groups), replace=False)
mz_e, d_e = np.arange(50, 1001, 25.0), np.arange(-0.15, 0.451, 0.01); H = np.zeros((len(mz_e) - 1, len(d_e) - 1)); npk = 0
for gi in groups:
    t = pf.read_row_group(int(gi), columns=["ms2_mzs", "ms2_normalized_intensities"]).to_pandas()
    mz = np.concatenate(t.ms2_mzs.values); it = np.concatenate(t.ms2_normalized_intensities.values)
    k = it >= 0.05; mz, it = mz[k], it[k]; d = mz - np.floor(mz + 0.05)
    h, _, _ = np.histogram2d(mz, d, bins=[mz_e, d_e], weights=it); H += h; npk += len(mz)
out["defect_mz_edges"] = mz_e.tolist(); out["defect_edges"] = d_e.tolist(); out["defect_hist"] = H.tolist(); out["defect_n_peaks"] = int(npk)
inside = ((d_e[:-1] >= -0.05) & (d_e[1:] <= 0.25)); out["defect_share_in_band"] = float(H[:, inside].sum() / H.sum())
json.dump(out, open("artifacts/release_eda.json", "w"))
print("saved; defect share in [-0.05, 0.25):", round(out["defect_share_in_band"], 3), "| peaks used:", npk)
