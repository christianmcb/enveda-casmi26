"""The v0.1 baseline (Ridge on 1 Da bins) rebuilt as a module, so every later problem can be judged on the
same 230k-pool benchmark without a Kaggle session.  Behaviour is deliberately identical to the notebook
(docs/CONTEXT.md section 5), including its quirks (see `ce_multi` below). Needs artifacts/ from
src/build_canon_cache.py and the decoy files dec_*.npy from v0.1.
"""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.linalg
import scipy.sparse as sp
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit

RDLogger.DisableLog("rdApp.*")
TRAIN_PATH = "data/train.parquet"
ART = "artifacts"
SMILES_COL = "normalized_smiles"
MAX_MZ, NB = 1000, 1000
# v0.1 ADDUCT table with KEEP_DIMERS=False: (n_molecules, shift)
ADDUCT = {"[M+H]+": (1, 1.007276), "[M-H]-": (1, -1.007276), "[M+Na]+": (1, 22.989218),
          "[M+NH4]+": (1, 18.033823), "[M+CH2O2-H]-": (1, 44.997655)}


def read_rows(row_ids, columns):
    """Fetch `columns` of train.parquet for the given file row positions, as a DataFrame in the order given."""
    pf = pq.ParquetFile(TRAIN_PATH)
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    row_ids = np.asarray(row_ids)
    parts = []
    for g in range(pf.num_row_groups):
        m = (row_ids >= starts[g]) & (row_ids < starts[g + 1])
        if not m.any():
            continue
        t = pf.read_row_group(g, columns=columns).take(row_ids[m] - starts[g]).to_pandas()
        t.index = np.flatnonzero(m)
        parts.append(t)
    return pd.concat(parts).sort_index().reset_index(drop=True)


def make_split():
    """Steps 3b-5 of notebook cell 23, from the cached canonicalisation."""
    sub = pd.read_parquet(f"{ART}/sub_rows_v1.parquet")
    cache = pd.read_parquet(f"{ART}/canon_cache_v1.parquet")
    sub = (sub.merge(cache, left_on=SMILES_COL, right_on="smiles", how="left")
           .dropna(subset=["key", "scaffold"]).reset_index(drop=True))
    sub = sub.groupby("key").head(3).reset_index(drop=True)
    sub["scaffold"] = sub.groupby("key").scaffold.transform(lambda s: s.mode().iloc[0])
    empty = sub.scaffold == ""
    sub["is_acyclic"] = empty
    sub.loc[empty, "scaffold"] = "acyclic_" + sub.loc[empty, "key"]       # each acyclic molecule = own group
    tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(sub, groups=sub.scaffold))
    tr, va = sub.iloc[tr_i].reset_index(drop=True), sub.iloc[va_i].reset_index(drop=True)
    va = va[~va.key.isin(set(tr.key))].reset_index(drop=True)
    assert len(set(tr.key) & set(va.key)) == 0, "train/val share molecules"
    return sub, tr, va


def _ce(x):
    """v0.1: lists -> mean + flag. NOTE the flag is 1 for ANY list, and train stores every CE as a list,
    so `ce_multi` is effectively 'has a CE value' (not 'stepped'). Kept as-is for v0.1 fidelity."""
    if x is None:
        return np.nan, 0.0
    a = np.asarray(x, float)
    return (np.nan if a.size == 0 else float(np.nanmean(a))), 1.0


def load_spectra(d):
    r = read_rows(d["row"].values, ["ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev",
                                    "instrument_type", "ionization_mode"])
    d = d.reset_index(drop=True).copy()
    d["instrument_type"], d["ionization_mode"] = r.instrument_type.values, r.ionization_mode.values
    ce = [_ce(x) for x in r.collision_energy_ev]
    d["collision_energy_ev"], d["ce_multi"] = [c[0] for c in ce], np.float32([c[1] for c in ce])
    d["neutral_mass"] = (d.precursor_mz - d.adduct.map(lambda a: ADDUCT[a][1])) / d.adduct.map(lambda a: ADDUCT[a][0])
    return d, r.ms2_mzs.values, r.ms2_normalized_intensities.values


def spec_vec(mz, it):
    v = np.zeros(NB, np.float32)
    idx = (np.asarray(mz) / 1.0).astype(int)
    ok = (idx >= 0) & (idx < NB)
    np.add.at(v, idx[ok], np.sqrt(np.asarray(it)[ok]))
    return v / (np.linalg.norm(v) + 1e-9)


def loss_vec(mz, it, precursor_mz):
    """P4a: neutral-loss axis. loss = precursor m/z - fragment m/z, 1 Da bins, sqrt(intensity), unit length.
    Peaks at/above the precursor (loss < 1 Da: the precursor itself, isotopes, noise) are ignored."""
    loss = precursor_mz - np.asarray(mz)
    v = np.zeros(NB, np.float32)
    idx = loss.astype(int)
    ok = (loss >= 1.0) & (idx < NB)
    np.add.at(v, idx[ok], np.sqrt(np.asarray(it)[ok]))
    return v / (np.linalg.norm(v) + 1e-9)


def sparse_block(axes, its, bin_da, nb, eps=1e-6):
    """P4b: one block of binned peaks as a sparse matrix. `axes` = per-spectrum peak positions (fragment m/z, or loss with
    invalid peaks set to -1). Same recipe as spec_vec/loss_vec (sqrt intensity, unit length) at any bin width.
    eps guards float noise at bin edges (285.3 / 0.1 = 2852.9999999999995); eps=0 reproduces the dense functions exactly."""
    r_, c_, v_ = [], [], []
    for r, (a, i) in enumerate(zip(axes, its)):
        a = np.asarray(a, float)
        idx = np.floor(a / bin_da + eps).astype(int)
        ok = (a >= 0) & (idx < nb)
        if not ok.any():
            continue
        u, inv = np.unique(idx[ok], return_inverse=True)
        v = np.bincount(inv, weights=np.sqrt(np.asarray(i, float)[ok]))
        v /= np.linalg.norm(v) + 1e-9
        r_.append(np.full(len(u), r)); c_.append(u); v_.append(v)
    return sp.csr_matrix((np.concatenate(v_).astype(np.float32), (np.concatenate(r_), np.concatenate(c_))),
                         shape=(len(axes), nb))


DEFECT_LO, DEFECT_HI = -0.05, 0.25       # fragment mass defects live in this band (P4b diagnostic figure)


def defect_block(axes, its, w=0.02, nb_nominal=1000, eps=1e-6):
    """P4c: nominal-mass x mass-defect sub-bins. Per peak at position x: nominal n = floor(x + 0.05) (so 108.9985 and 109.0012
    share n=109), defect d = x - n. d in [-0.05, 0.25) goes to one of K = round(0.30 / w) sub-bins of width w; any other d goes to
    one overflow column per nominal mass. Columns per nominal mass = K + 1. Same weighting as the other blocks."""
    K = int(round((DEFECT_HI - DEFECT_LO) / w))
    per = K + 1
    r_, c_, v_ = [], [], []
    for r, (a, i) in enumerate(zip(axes, its)):
        a = np.asarray(a, float)
        n = np.floor(a + 0.05 + eps).astype(int)
        d = a - n
        ok = (a >= 0) & (n < nb_nominal)
        if not ok.any():
            continue
        sub = np.where((d >= DEFECT_LO) & (d < DEFECT_HI), np.floor((d - DEFECT_LO) / w + eps).astype(int).clip(0, K - 1), K)
        col = (n * per + sub)[ok]
        u, inv = np.unique(col, return_inverse=True)
        v = np.bincount(inv, weights=np.sqrt(np.asarray(i, float)[ok]))
        v /= np.linalg.norm(v) + 1e-9
        r_.append(np.full(len(u), r)); c_.append(u); v_.append(v)
    return sp.csr_matrix((np.concatenate(v_).astype(np.float32), (np.concatenate(r_), np.concatenate(c_))),
                         shape=(len(axes), nb_nominal * per))


class NormalEqRidge:
    """Ridge(alpha, fit_intercept=True) for SPARSE X via the normal equations (sklearn's sparse path is per-target and far
    too slow for 1,024 targets). Columns that are empty in the training data are dropped (no effect on predictions).
    G = X'X is accumulated in dense chunks once, so refitting at another alpha is just one more solve."""
    def __init__(self, alpha=1.0, chunk=2000):
        self.alpha, self.chunk = alpha, chunk

    def fit(self, X, Y):
        X = sp.csr_matrix(X)
        self.keep = np.flatnonzero(np.asarray((X != 0).sum(0)).ravel() > 0)
        X = X[:, self.keep].astype(np.float64)          # float64 throughout: mixing precisions breaks positive-definiteness
        n, p = X.shape
        self.consume = p * p * 8 > 5e9                  # too big to keep a pristine copy next to the factorisation
        print(f"  NormalEqRidge: {n:,} rows x {p:,} non-empty columns, G = {p * p * 8 / 1e9:.1f} GB"
              f"{' (consumed by the solve: no refit)' if self.consume else ''}", flush=True)
        G = np.zeros((p, p), order="F")
        for i in range(0, n, self.chunk):               # symmetric rank-k update of the LOWER triangle, in place (half the flops, no p x p temporary)
            G = scipy.linalg.blas.dsyrk(1.0, X[i:i + self.chunk].toarray(), trans=1, beta=1.0, c=G, lower=1, overwrite_c=1)
        self.mu = np.asarray(X.mean(0)).ravel()
        for i in range(0, p, 2000):                     # centre in place (lower triangle is all that is used)
            G[i:i + 2000] -= n * np.outer(self.mu[i:i + 2000], self.mu)
        self.Gc = G
        # X'Y and the target mean accumulated in row chunks, so Y may stay uint8/float32 (a float64 copy is 5+ GB at 650k rows)
        XtY, ysum = np.zeros((p, Y.shape[1])), np.zeros(Y.shape[1])
        for i in range(0, n, 20000):
            Yc = np.asarray(Y[i:i + 20000], np.float64)
            XtY += np.asarray(X[i:i + 20000].T @ Yc); ysum += Yc.sum(0)
        self.ymu = ysum / n
        self.XtY = XtY - n * np.outer(self.mu, self.ymu)
        return self.refit(self.alpha)

    def refit(self, alpha):
        self.alpha = alpha
        if self.Gc is None:
            raise RuntimeError("Gram matrix was consumed; refit this configuration from scratch")
        A = self.Gc if self.consume else self.Gc.copy(order="F")
        A[np.diag_indices_from(A)] += alpha
        c = scipy.linalg.cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
        self.W = scipy.linalg.cho_solve(c, self.XtY, check_finite=False).astype(np.float32)
        del A, c
        if self.consume:
            self.Gc = None
        self.b = (self.ymu - self.mu @ self.W).astype(np.float32)
        return self

    def predict(self, X):
        return (sp.csr_matrix(X)[:, self.keep] @ self.W) + self.b


class Features:
    """v0.1 features. neutral_loss=True appends a second block (P4a). frag_bin / loss_bin != 1.0 switch to the sparse
    fine-bin representation (P4b); the defaults reproduce v0.1/v0.2a exactly."""
    def __init__(self, tr, neutral_loss=False, fragment=True, frag_bin=1.0, loss_bin=1.0, frag_defect=None, loss_defect=None):
        """frag_defect / loss_defect: sub-bin width w (Da) to use the P4c nominal x defect scheme for that block instead of uniform bins."""
        self.neutral_loss, self.fragment = neutral_loss, fragment
        self.frag_bin, self.loss_bin = frag_bin, loss_bin
        self.frag_defect, self.loss_defect = frag_defect, loss_defect
        self.sparse = frag_bin != 1.0 or loss_bin != 1.0 or frag_defect is not None or loss_defect is not None
        self.top_inst = tr.instrument_type.value_counts().head(8).index
        self.cols = self._meta(tr).columns
        self.ce_med = tr.collision_energy_ev.median()

    def _meta(self, d):
        m = d[["ionization_mode", "instrument_type", "adduct"]].copy()
        m["instrument_type"] = m.instrument_type.where(m.instrument_type.isin(self.top_inst), "other")
        return pd.get_dummies(m)

    def build_sparse(self, d, mzs, its):
        blocks = []
        if self.fragment:
            blocks.append(defect_block(mzs, its, self.frag_defect) if self.frag_defect
                          else sparse_block(mzs, its, self.frag_bin, int(round(MAX_MZ / self.frag_bin))))
        if self.neutral_loss:
            ax = [np.where(p - np.asarray(m) >= 1.0, p - np.asarray(m), -1.0) for m, p in zip(mzs, d.precursor_mz.values)]
            blocks.append(defect_block(ax, its, self.loss_defect) if self.loss_defect
                          else sparse_block(ax, its, self.loss_bin, int(round(MAX_MZ / self.loss_bin))))
        M = self._meta(d).reindex(columns=self.cols, fill_value=0).values.astype(np.float32)
        num = np.c_[d.neutral_mass / MAX_MZ, d.collision_energy_ev.fillna(self.ce_med) / 100, d.ce_multi].astype(np.float32)
        return sp.hstack(blocks + [sp.csr_matrix(np.hstack([num, M]))], format="csr")

    def build(self, d, mzs, its):
        if self.sparse:
            return self.build_sparse(d, mzs, its)
        blocks = []
        if self.fragment:
            blocks.append(np.stack([spec_vec(m, i) for m, i in zip(mzs, its)]))
        if self.neutral_loss:
            blocks.append(np.stack([loss_vec(m, i, p) for m, i, p in zip(mzs, its, d.precursor_mz.values)]))
        S = np.hstack(blocks)
        M = self._meta(d).reindex(columns=self.cols, fill_value=0).values.astype(np.float32)
        num = np.c_[d.neutral_mass / MAX_MZ, d.collision_energy_ev.fillna(self.ce_med) / 100, d.ce_multi].astype(np.float32)
        return np.hstack([S, num, M])


unit = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def molecule_table(sub):
    """One row per molecule: Morgan-1024 fingerprint (unit length), exact mass, key."""
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    mols_u = sub.drop_duplicates("key").reset_index(drop=True)
    objs = [Chem.MolFromSmiles(s) for s in mols_u[SMILES_COL]]
    FP = np.stack([fpgen.GetFingerprintAsNumPy(m) for m in objs]).astype(np.float32)
    mass = np.array([Descriptors.ExactMolWt(m) for m in objs])
    return mols_u, FP, mass


def load_pool(mols_u, FP, mass):
    """The 229,964-molecule benchmark pool: the 29,964 oracle molecules + 200k decoys saved by v0.1."""
    keys = np.concatenate([mols_u.key.values, np.load("dec_keys.npy")])
    fp_n = np.vstack([unit(FP), np.load("dec_fp_n.npy")])
    m = np.concatenate([mass, np.load("dec_mass.npy")])
    return keys, fp_n, m


def load_data():
    """Split + peak arrays + fingerprints, no features yet (so several feature sets can share one load)."""
    sub, tr, va = make_split()
    print(f"split: {len(tr):,} train | {len(va):,} val  (v0.1: 61,701 / 14,547)")
    tr, mz_t, it_t = load_spectra(tr)
    va, mz_v, it_v = load_spectra(va)
    mols_u, FP, mass = molecule_table(sub)
    row_of = pd.Series(np.arange(len(mols_u)), index=mols_u.key)
    return dict(sub=sub, tr=tr, va=va, mz_t=mz_t, it_t=it_t, mz_v=mz_v, it_v=it_v,
                mols_u=mols_u, FP=FP, mass=mass, Ytr=FP[row_of[tr.key].values])


def fit_ridge(D, neutral_loss=False, alpha=1.0, fragment=True, cache=None, frag_bin=1.0, loss_bin=1.0, solver="sklearn",
              frag_defect=None, loss_defect=None):
    """cache: optional dict; feature matrices are reused across alphas for the same configuration.
    solver='normal_eq' uses NormalEqRidge (needed for sparse fine bins; same maths as sklearn Ridge)."""
    feat = Features(D["tr"], neutral_loss=neutral_loss, fragment=fragment, frag_bin=frag_bin, loss_bin=loss_bin,
                    frag_defect=frag_defect, loss_defect=loss_defect)
    ck = (neutral_loss, fragment, frag_bin, loss_bin, solver, frag_defect, loss_defect)
    mk = (lambda: NormalEqRidge(alpha)) if solver == "normal_eq" else (lambda: Ridge(alpha=alpha))
    if cache is not None and ck in cache:
        return dict(feat=feat, **{k: cache[ck][k] for k in ("Xtr", "Xva")}, reg=cache[ck]["reg"].refit(alpha)
                    if solver == "normal_eq" else mk().fit(cache[ck]["Xtr"], D["Ytr"]))
    Xtr, Xva = feat.build(D["tr"], D["mz_t"], D["it_t"]), feat.build(D["va"], D["mz_v"], D["it_v"])
    reg = mk().fit(Xtr, D["Ytr"])
    if cache is not None:
        cache[ck] = dict(Xtr=Xtr, Xva=Xva, reg=reg)
    return dict(feat=feat, Xtr=Xtr, Xva=Xva, reg=reg)


def build_v1():
    D = load_data()
    M = fit_ridge(D)
    return dict(sub=D["sub"], tr=D["tr"], va=D["va"], Xtr=M["Xtr"], Xva=M["Xva"], reg=M["reg"],
                mols_u=D["mols_u"], FP=D["FP"], mass=D["mass"])
