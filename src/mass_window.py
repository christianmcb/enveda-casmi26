"""Neutral-mass maths and the candidate mass window (P3).

Findings this module encodes (docs/EXPERIMENT_LOG.md, P3):
  * `precursor_error_ppm` in train.parquet is UNSIGNED and was computed with the hydrogen ATOM mass, so it
    hides the real calibration offset. Recompute the SIGNED error from `molecular_formula` instead.
  * On the test-like instrument (enveda-180, timsTOF) the offset is per adduct ([M+H]+ ~ +2.2 ppm,
    [M-H]- ~ +0.8 ppm) with a robust spread of only ~0.9 ppm. A window centred on that offset is much tighter
    than one centred on zero.
"""
import re

import numpy as np
import pandas as pd

# neutral-mass shifts (Da): ion m/z = M + shift. Includes the two extra adducts that occur in test.
SHIFT = {"[M+H]+": 1.007276, "[M-H]-": -1.007276, "[M+Na]+": 22.989218, "[M+NH4]+": 18.033823,
         "[M+CH2O2-H]-": 44.997655, "[M+K]+": 38.963158, "[M+Cl]-": 34.969402}

_FORMULA = re.compile(r"([A-Z][a-z]?)(\d*)")
_cache = {}


def formula_mass(formula):
    """Monoisotopic mass of a neutral molecular formula such as 'C20H28N2O3' (NaN if it cannot be parsed)."""
    from rdkit import Chem
    if formula in _cache:
        return _cache[formula]
    pt = Chem.GetPeriodicTable()
    try:
        if not re.fullmatch(r"([A-Z][a-z]?\d*)+", formula):
            raise ValueError
        m = sum(pt.GetMostCommonIsotopeMass(el) * (int(n) if n else 1) for el, n in _FORMULA.findall(formula))
    except Exception:
        m = np.nan
    _cache[formula] = m
    return m


def neutral_mass(precursor_mz, adduct):
    """Neutral mass implied by a monomer adduct (NaN for adducts not in SHIFT, e.g. dimers)."""
    return np.asarray(precursor_mz, float) - pd.Series(adduct).map(SHIFT).values


def signed_error_ppm(nm, true_mass):
    return (np.asarray(nm) - np.asarray(true_mass)) / np.asarray(true_mass) * 1e6


def fit_bias(err_ppm, adduct):
    """Per-adduct calibration offset (median signed error, ppm) and robust spread (1.4826*MAD)."""
    g = pd.DataFrame({"err": err_ppm, "adduct": adduct}).groupby("adduct").err
    return pd.DataFrame({"n": g.size(), "bias_ppm": g.median(),
                         "sd_ppm": g.apply(lambda s: 1.4826 * np.median(np.abs(s - s.median())))})


def halfwidth_by_adduct(bias_tab, k=3.5, floor=3.0):
    """Per-adduct half-width (ppm): k robust sigmas, never below `floor`. Rare adducts ([M+NH4]+, [M+K]+) scatter more."""
    return (k * bias_tab.sd_ppm).clip(lower=floor).to_dict()


def window(nm, adduct, tol_ppm, bias=None):
    """Candidate-mass interval [lo, hi] for each query.

    bias=None reproduces the v0.1 policy (+-tol around the measured neutral mass).
    With a bias table (adduct -> ppm) the window is centred on the corrected mass nm / (1 + bias).
    tol_ppm is a scalar, or a dict adduct -> ppm (adducts missing from it get the widest value).
    """
    nm = np.asarray(nm, float)
    if bias is not None:
        nm = nm / (1 + pd.Series(adduct).map(bias).fillna(0.0).values * 1e-6)
    if isinstance(tol_ppm, dict):
        tol_ppm = pd.Series(adduct).map(tol_ppm).fillna(max(tol_ppm.values())).values
    return nm * (1 - tol_ppm * 1e-6), nm * (1 + tol_ppm * 1e-6)
