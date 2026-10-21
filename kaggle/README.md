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

## Fast iteration: freeze once, then infer in minutes (added 2026-10-21)
A full training notebook takes ~2 h (canonicalisation 22 min, features, ~1 h deep MLP), and almost none of it depends on the candidate pool or the popularity prior. So:
1. **Train + freeze (once per model change).** Run `casmi_submission_deep.ipynb` as usual. With `FREEZE = True` (default) it writes `/kaggle/working/frozen/`: `deep_best.pt` (weights), `keep.npy` (input columns), `frozen_meta.json` (feature settings, calibration offsets, sizes), `train_library.parquet` + `train_library_fp.npy` (the TRAINING part of the library; COCONUT and the popularity table are read from their own datasets). About 0.6 GB.
   - **Skip the 2 h for the run that already exists:** attach that earlier notebook's output as an input and set `RESTORE_CACHE_DIR = "auto"` (first config cell). The notebook copies `canon_cache.parquet`, `X_train.npz`, `deep_pred.npz` and `deep_best.pt` into `/kaggle/working`, skips canonicalisation/features/training and only freezes (~10-20 min). Valid only if the training data and the training code are unchanged.
2. **Make the frozen files an input.** On the finished run: Output tab -> New Dataset -> select the `frozen` folder -> private. (Keep it private; it contains the trained model and the training molecules' SMILES.)
3. **Infer.** Import `casmi_infer.ipynb`, attach: the competition data, the frozen dataset, the COCONUT cache dataset, and (if `POP_WEIGHT > 0`) `dmitriigluzdov/casmi26-pubchem-popularity-prior`. No GPU needed (CPU sessions also queue less). Edit only `POP_WEIGHT` / `EXTRA_POOL_DIR` in the config cell; the run takes minutes. Commit it and submit its `submission.csv`.
4. The inference notebook is also the safe choice for the hidden-test rerun: it does not retrain, so it cannot run into the 9 h limit.
Local equivalence test: a frozen DEBUG run + `casmi_infer` must write the same `submission_deep_debug.csv` as the training run (checked 2026-10-21, see docs/EXPERIMENT_LOG).

## Running casmi_infer with the PubChem channel (P7d, added 2026-10-21)
Edit ONLY the block `SETTINGS: THE ONLY LINES TO EDIT` at the top of the config cell: `POP_WEIGHT`, `USE_TIER`, `TIER_LIB_BONUS` (and leave `TIER_WORKERS = 4`). The log prints a `SETTINGS:` line with what was actually used. The values to use are in docs/EXPERIMENT_LOG.md (entry "P7d").
- `USE_TIER = False` reproduces the current 0.146 submission (w = 0.15).
- `USE_TIER = True` adds PubChem structures from each query's mass window (fingerprinted on the fly, ~0.5 s per molecule locally), scored with the same prior; `TIER_LIB_BONUS` is a source prior in z-score units that favours train + COCONUT candidates.
- Attach the same inputs as before; the tier arrays are inside the popularity dataset (`dmitriigluzdov/casmi26-pubchem-popularity-prior`). The first cells check that the six tier files exist and stop with a clear message if not. A GPU is not needed.
- A commit run does NOT use a daily submission; only the Submit click does. Test `USE_TIER = True` with a commit first (check runtime, no errors, the `share of PubChem-tier guesses` line), then submit the version with the final settings.

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
