"""The 230k-pool benchmark as a reusable object: score ANY model's predicted fingerprints on both reporting slices.

Slices (CLAUDE.md rule 5, approved 2026-10-20):
  mixed_2000      the v0.1 evaluation mix (2,000 seeded val spectra, all sources), v0.1 window: +-10 ppm around 0
  enveda180       all enveda-180 val spectra (test-like instrument/processing), P3 window (offset-centred, per-adduct width)  <- primary
  enveda180_10ppm same spectra with the v0.1 window, for continuity with older numbers
Offsets are fitted on the TRAINING split only. 'random' is the exact expected score of random order inside the window.
"""
import numpy as np
import pandas as pd

import mass_window as mw
import v1_pipeline as v1


class Bench:
    def __init__(self, D, all_val=False, extra=None):
        tr, va = D["tr"], D["va"]
        self.va, self.D = va, D
        self.keys, self.fp, self.mass = v1.load_pool(D["mols_u"], D["FP"], D["mass"])
        if extra is not None:          # extra = (keys, unit-length fingerprints, masses): more decoy candidates (e.g. COCONUT), deduplicated by key
            ek, efp, em = extra
            keep = ~np.isin(ek, self.keys)
            self.keys = np.concatenate([self.keys, ek[keep]]); self.fp = np.vstack([self.fp, efp[keep]]); self.mass = np.concatenate([self.mass, em[keep]])
            print(f"pool: +{int(keep.sum()):,} extra candidates -> {len(self.keys):,}")
        self.order = np.argsort(self.mass)
        self.ms = self.mass[self.order]
        self.inv = np.empty(len(self.order), int)
        self.inv[self.order] = np.arange(len(self.order))
        self.key_pos = pd.Series(np.arange(len(self.keys)), index=self.keys)
        self.H = np.concatenate([[0.0], np.cumsum(1 / np.arange(1, 26))])
        # leak-free calibration offsets from the training split (true mass from the SMILES)
        t180 = tr[tr.ingest_lib == "enveda-180"]
        true_m = pd.Series(D["mass"], index=D["mols_u"].key)[t180.key].values
        self.bias_tab = mw.fit_bias(mw.signed_error_ppm(t180.neutral_mass.values, true_m), t180.adduct.values)
        self.BIAS, self.HW = self.bias_tab.bias_ppm.to_dict(), mw.halfwidth_by_adduct(self.bias_tab)
        vi = np.random.default_rng(42).choice(len(va), size=min(2000, len(va)), replace=False)
        e180 = np.flatnonzero((va.ingest_lib == "enveda-180").values)
        self.slices = {"mixed_2000": (vi, 10.0, None), "enveda180": (e180, self.HW, self.BIAS),
                       "enveda180_10ppm": (e180, 10.0, None)}
        if all_val:      # every val spectrum with the v0.1 window; used for per-source and novelty breakdowns
            self.slices["all_val"] = (np.arange(len(va)), 10.0, None)
        self.FPu = v1.unit(D["FP"])
        self.row_of = pd.Series(np.arange(len(D["mols_u"])), index=D["mols_u"].key)
        self.per_query = {}          # (name, slice) -> per-query DataFrame, for paired bootstrap comparisons

    def _evaluate(self, Pq, true_keys, lo, hi, binary=False):
        tp = self.inv[self.key_pos[true_keys].values]
        out = np.zeros((len(lo), 4))
        for i in range(len(lo)):
            a, b = np.searchsorted(self.ms, [lo[i], hi[i]])
            n, inside = b - a, a <= tp[i] < b
            rr_m = rr_r = 0.0
            if inside:
                idx = self.order[a:b]
                s = (self.fp_bin[idx].astype(np.float32) if binary else self.fp[idx]) @ Pq[i]
                rank = 1 + int((s > s[np.flatnonzero(idx == self.order[tp[i]])[0]]).sum())
                rr_m = 1 / rank if rank <= 25 else 0.0
                rr_r = self.H[min(n, 25)] / n
            out[i] = (n, inside, rr_m, rr_r)
        return pd.DataFrame(out, columns=["cands", "inside", "rr_model", "rr_random"])

    def score(self, name, Pva, rule="cosine", eps=1e-4):
        """Pva: predicted fingerprints for ALL val rows. Returns one row per slice.
        rule='cosine' (v0.1 default): cosine between the prediction and each candidate fingerprint.
        rule='loglik': Bernoulli log-likelihood of the candidate's bits under the predicted probabilities, which for ranking
        reduces to sum(candidate_bit * logit(p)); p is clipped to [eps, 1-eps]."""
        Pva = np.asarray(Pva, np.float32)
        binary = rule == "loglik"
        if binary:
            if not hasattr(self, "fp_bin"):
                self.fp_bin = (self.fp > 0).astype(np.uint8)          # pool fingerprints are stored unit-normalised; > 0 recovers the bits
            pc = np.clip(Pva, eps, 1 - eps); Pq_all = np.log(pc) - np.log1p(-pc)
        else:
            Pq_all = v1.unit(Pva)
        rows = []
        for sname, (ix, tol, bias) in self.slices.items():
            q = self.va.iloc[ix]
            lo, hi = mw.window(q.neutral_mass.values, q.adduct.values, tol, bias)
            r = self._evaluate(Pq_all[ix], q.key.values, lo, hi, binary=binary)
            self.per_query[(name, sname)] = r.assign(key=q.key.values)
            truth = self.FPu[self.row_of[q.key].values]
            rows.append(dict(version=name, slice=sname, n=len(r), model=r.rr_model.mean(), random=r.rr_random.mean(),
                             model_minus_random=r.rr_model.mean() - r.rr_random.mean(), cands_mean=r.cands.mean(),
                             cands_median=r.cands.median(), miss_pct=(1 - r.inside.mean()) * 100,
                             fp_cosine=float((v1.unit(Pva)[ix] * truth).sum(1).mean()), rule=rule))   # pool-independent model quality
        return pd.DataFrame(rows)

    def paired_diff(self, a, b, sname, n_boot=2000, seed=0):
        """Mean model-score difference b - a on a slice with a molecule-clustered paired bootstrap (95% CI)."""
        ra, rb = self.per_query[(a, sname)], self.per_query[(b, sname)]
        d = (rb.rr_model - ra.rr_model).groupby(ra.key.values).agg(["sum", "size"])   # spectra of one molecule move together
        rng = np.random.default_rng(seed)
        idx = rng.integers(len(d), size=(n_boot, len(d)))
        boots = d["sum"].values[idx].sum(1) / d["size"].values[idx].sum(1)
        return float((rb.rr_model - ra.rr_model).mean()), tuple(np.percentile(boots, [2.5, 97.5]))
