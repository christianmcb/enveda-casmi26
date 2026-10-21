# Kaggle submission notebook

One notebook version = one `submission.csv`, so there is one notebook per model (each writes `/kaggle/working/submission.csv`, the file Kaggle submits):
- `casmi_submission_baseline.ipynb`: Ridge baseline only (cheaper: no deep training).
- `casmi_submission_deep.ipynb`: deep residual MLP only.
- `casmi_submission_both.ipynb`: trains both and writes `submission.csv` from `FINAL_CHOICE` (dev/testing; the other model's CSV is kept in the output but cannot be submitted separately).
All three are generated from `work/kaggle_cells.py` (`python work/build_kaggle_notebook.py`); edit the source, not the .ipynb files.
Run each notebook separately (the caches in /kaggle/working are per notebook, so each recomputes canonicalisation and features, about 15-20 minutes).

## How to run it on Kaggle
1. Kaggle -> Code -> New Notebook -> File -> Import Notebook -> upload `casmi_submission_baseline.ipynb` (then repeat with `casmi_submission_deep.ipynb` as a second notebook).
2. Add the competition data (it appears under `/kaggle/input/...`; the notebook finds `train.parquet` itself).
3. Settings -> Accelerator -> **GPU** (T4 or P100). Internet can stay off. RDKit is not preinstalled: add `rdkit==2026.3.3` through the notebook's dependency manager (as in the v0.1 notebook), exactly as before; the first cell prints the version (the metric pins 2026.03.3).
4. **First run with `DEBUG = True`** (first code cell). It uses 6,000 molecules and 2 training epochs: a few minutes. It also writes a `submission.csv`, but that one comes from a tiny library, so do not submit it. Check that every cell finishes and the holdout-sanity lines print non-zero scores.
5. Set `DEBUG = False` and run all. Expect roughly 1.5-3 hours (unmeasured on Kaggle; on a Mac with 8 cores + Apple GPU: canonicalisation ~5 min, features ~4 min, Ridge ~10-25 min, deep MLP ~50 min). Timings are printed.
6. Submit: "Save Version" -> "Save & Run All (Commit)". The notebook writes `/kaggle/working/submission.csv` which is the file Kaggle uses when you submit the notebook version.

## What to expect from the score (important)
- The visible `test.parquet` is a **placeholder drawn from the training spectra** (all 1,213 spectra have exact twins in train). Its score, and the "how many test molecules are in train" probe printed at the end, mean nothing. Only the hidden-test rerun scores.
- The candidate library is every training molecule (~254k) **plus ~432k COCONUT natural products** (`EXTRA_POOL_DIR = "auto"`). COCONUT-only molecules have no spectra, so their value cannot be validated locally; the local benchmark shows only the COST of the bigger pool (about -0.11 MRR on NP-like validation molecules, docs/EXPERIMENT_LOG.md "P7a"). The leaderboard score is the first measurement of the gain.
- Set `EXTRA_POOL_DIR = None` for a train-only library (expected near 0 on novel molecules).

## Attach the COCONUT cache (needed once)
1. Locally run `python src/build_coconut_cache.py` (makes `artifacts/coconut_v1.parquet` and `artifacts/coconut_fp_v1.npy`; ~30 min).
2. Kaggle -> Datasets -> New Dataset -> upload those two files, keep it **private** (derived from prvsiyan/coconut-casmi26-candidates, CC BY-NC-SA 4.0; credit the author in the write-up).
3. In the notebook: Add Input -> your dataset. The notebook finds `coconut_v1.parquet` under `/kaggle/input` by itself and prints `+ N COCONUT candidates`. If it prints the WARNING instead, the dataset is not attached.

## What was tested locally (not on Kaggle)
- The notebook's ranking, windowing and pooling code reproduces the project benchmark on identical inputs (`work/test_kaggle_rank.py`: per-spectrum MRR 0.6715 vs 0.6706; pooled 0.7328 vs 0.7314).
- A full end-to-end run in DEBUG mode (`CASMI_DEBUG=1 CASMI_COMP_DIR=./data`) on the real files: all cells run, both CSVs pass the competition's rejection rules (400 ids in sample order, no nulls, <= 25 guesses).
- NOT tested: a full-size run (hours), Kaggle's RDKit/torch versions, Kaggle GPU speed and the 9-hour limit.

## Method summary
Features: fragment peaks + neutral losses (precursor m/z - fragment m/z), each binned as nominal mass x 0.02 Da mass-defect sub-bins (sparse), plus acquisition metadata.
Baseline: Ridge (alpha 1) on the v0.1 subsample (<= 5,000 molecules per source, <= 3 spectra each). Deep: residual MLP (1 linear + 4 pre-norm residual blocks, 1,024 sigmoid outputs, BCE) on every molecule (<= 3 spectra each), early stopping on a scaffold-grouped inner holdout.
Per test molecule: mean predicted fingerprint of its spectra; mass window = intersection of the spectra's windows, each centred on the per-adduct instrument offset (enveda-180 calibration) with half-width max(3 ppm, 3.5 sigma); rank library molecules in the window by cosine similarity; top 25.
Both models exclude a 3% scaffold-grouped holdout used for the sanity score.
