"""Cache COCONUT as extra candidate molecules in OUR representation (v0.4a, Step 2 of the external-pool plan).

Source: Kaggle dataset prvsiyan/coconut-casmi26-candidates (CC BY-NC-SA 4.0), downloaded to data/external/coconut/.
We use only its SMILES (coco_meta.pkl). Its fingerprints (coco_fp.npy) use a different bit layout and are NOT used.
For each SMILES: canonical key (InChIKey14 of the RDKit tautomer-canonicalised molecule = the metric recipe, same as
src/build_canon_cache.py), exact mass, Morgan radius-2 1,024-bit fingerprint (same generator as src/v1_pipeline.py).

Writes
    artifacts/coconut_v1.parquet   smiles, stored_key (source's InChIKey14), key (metric recipe), mass   (rows aligned with the .npy)
    artifacts/coconut_fp_v1.npy    uint8 (n, 1024)
Then prints the overlap with train/val, which decides how much a COCONUT pool can help.

Run from the repo root:  python src/build_coconut_cache.py        (use RDKit 2026.03.3; ~20-40 min on 8 cores)
The pickle holds only numpy arrays (checked with pickletools: only numpy reconstruct globals); it is still third-party data.
"""
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

SRC = "data/external/coconut/coco_meta.pkl"
OUT = "artifacts"


def _init():
    global _taut, _fpgen
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    _taut = rdMolStandardize.TautomerEnumerator()
    _fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def work(smi):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None, np.nan, None
        key = Chem.MolToInchiKey(_taut.Canonicalize(m))[:14]
        return key, Descriptors.ExactMolWt(m), np.packbits(_fpgen.GetFingerprintAsNumPy(m))
    except Exception:
        return None, np.nan, None


if __name__ == "__main__":
    import rdkit
    print("RDKit", rdkit.__version__, "(competition pins 2026.03.3)")
    with open(SRC, "rb") as f:
        meta = pickle.load(f)                                  # dict of numpy arrays: keys, smiles, nbits
    smiles, stored = list(meta["smiles"]), list(meta["keys"])
    print(f"COCONUT rows: {len(smiles):,}")

    with Pool(os.cpu_count(), initializer=_init) as pool:
        res = list(tqdm(pool.imap(work, smiles, chunksize=200), total=len(smiles)))
    ok = np.array([r[0] is not None for r in res])
    print(f"failed to parse/canonicalise: {(~ok).sum():,}")
    df = pd.DataFrame({"smiles": smiles, "stored_key": stored, "key": [r[0] for r in res], "mass": [r[1] for r in res]})
    fp = np.zeros((len(df), 1024), np.uint8)
    for i, r in enumerate(res):
        if r[2] is not None:
            fp[i] = np.unpackbits(r[2])
    df, fp = df[ok].reset_index(drop=True), fp[ok]
    n0 = len(df)
    first = ~df.key.duplicated()                               # one row per canonical key
    df, fp = df[first].reset_index(drop=True), fp[first.values]
    print(f"unique canonical keys: {len(df):,} (dropped {n0 - len(df):,} duplicates)")
    print("stored key == metric key for", f"{(df.stored_key == df.key).mean():.1%}", "of rows (they can differ: tautomers)")
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(f"{OUT}/coconut_v1.parquet")
    np.save(f"{OUT}/coconut_fp_v1.npy", fp)

    # ---- overlap report ----
    ck = set(df.key)
    known = {}
    for name in ("canon_cache_v1.parquet", "scale/canon_extra.parquet"):
        t = pd.read_parquet(f"{OUT}/{name}")
        known[name] = set(t.key.dropna())
        print(f"{name}: {len(known[name]):,} keys, {len(ck & known[name]):,} in COCONUT ({len(ck & known[name]) / len(known[name]):.1%})")
    import sys
    sys.path.insert(0, "src")
    import v1_pipeline as v1
    sub, tr, va = v1.make_split()
    for name, d in (("val molecules (v0.1 split)", va), ("train molecules (v0.1 split)", tr)):
        k = set(d.key)
        print(f"{name}: {len(k):,}, in COCONUT {len(k & ck):,} ({len(k & ck) / len(k):.1%})")
    e = va[va.ingest_lib == "enveda-180"]
    print(f"  enveda-180 val molecules in COCONUT: {e.key.nunique() and len(set(e.key) & ck):,} of {e.key.nunique():,}")
    allk = known["canon_cache_v1.parquet"] | known["scale/canon_extra.parquet"]
    print(f"COCONUT molecules NOT in any cached train key: {len(ck - allk):,} of {len(ck):,} (new candidates)")
    m = df.mass.values
    print("COCONUT mass quantiles 1/50/99%:", np.percentile(m, [1, 50, 99]).round(1), " share in 157-1160 Da:",
          f"{((m >= 157) & (m <= 1160)).mean():.1%}")
