# Enveda CASMI 2026 — Molecule ID from MS/MS spectra

**Baseline, release v0.1** · Kaggle competition *Enveda CASMI 2026 – Molecule ID From Mass Spectra* · metric **MRR@25**

Given an LC-MS/MS spectrum of a natural product, predict its 2D structure (SMILES), including molecules that never appear in the training data. This repository is the first snapshot of a leakage-aware baseline: a spectrum → molecular-fingerprint model, a mass-window candidate filter, a validation setup that can be trusted, and the experiments that took the local benchmark from 0.17 to 0.67 MRR@25.

> **Read this first (honest status).** The first Kaggle submission (deep MLP, candidate library = training molecules only) scored **0.076** on the leaderboard. A train-only library cannot contain novel molecules, so most hidden-test molecules are unreachable regardless of model quality. The local benchmark numbers below measure *ranking quality when the true molecule is in the candidate pool*; they are **not** leaderboard predictions. Extending the candidate pool (COCONUT, PubChem) is the next step and is not part of this release's results. See [Limitations](#limitations).

---

## Contents
1. [Problem and metric](#problem-and-metric)
2. [Approach](#approach)
3. [Results at a glance](#results-at-a-glance)
4. [The v0.1 baseline in detail](#the-v01-baseline-in-detail)
5. [Improvements after v0.1](#improvements-after-v01)
6. [Limitations](#limitations)
7. [Repository layout](#repository-layout)
8. [Reproducing](#reproducing)
9. [Data, rules and licences](#data-rules-and-licences)
10. [Roadmap](#roadmap)

---

## Problem and metric
- **Input:** a tandem mass spectrum (list of fragment m/z + intensity), the precursor m/z and the adduct (e.g. `[M+H]+`), plus acquisition metadata. No formula or structure is given. Several spectra (different collision energies/adducts/polarities) can belong to one `molecule_id`.
- **Output:** up to 25 SMILES per `molecule_id`, best first, joined by `;`.
- **Metric:** MRR@25. Rank 1 scores 1.0, rank 2 scores 0.5, … rank 25 scores 0.04, a miss scores 0. A hit means the InChIKey14 (connectivity block) matches after RDKit tautomer canonicalisation (RDKit 2026.03.3); stereochemistry and tautomers are not penalised.
- **Training data:** ~2.54M spectra from open libraries and Enveda-180 (timsTOF); after cleaning, 1.73M spectra and 253,832 unique molecules.

## Approach
```
spectrum ──► sparse features ──► model ──► predicted Morgan fingerprint (1,024 bits)
                                                      │
precursor m/z + adduct ──► neutral mass ──► mass window ──► candidate molecules ──► rank by cosine ──► top 25
```
1. **Cleaning.** Keep known adducts, precursor 50–1500 m/z, recorded |error| ≤ 20 ppm; collapse array-valued collision energies; drop dimers.
2. **Neutral mass** = (precursor m/z − adduct shift) / n. The mass window is a free, strong filter on candidates.
3. **Features** (v0.1: 1 Da bins; final: fragment block + neutral-loss block, each in nominal-mass × mass-defect sub-bins, ~32k sparse columns).
4. **Targets:** Morgan fingerprint, radius 2, 1,024 bits. **Model:** Ridge (v0.1) → deep residual MLP (v0.3b).
5. **Candidates:** library molecules whose exact mass falls in the per-adduct, instrument-offset-corrected window (centred, half-width max(3 ppm, 3.5σ)).
6. **Ranking:** cosine similarity between predicted and candidate fingerprints. For a test molecule with several spectra, the predicted fingerprints are averaged and the mass windows intersected (**molecule-level pooling**, +0.03 to +0.06 MRR).

**Validation that avoids the usual traps.** 80/20 split grouped by Murcko scaffold computed with stereo removed; every acyclic molecule is its own group; any validation molecule whose canonical key is in train is hard-removed and the overlap is `assert`ed to be zero. Tautomer canonicalisation is cached. Seeds are fixed (42). Scores are reported on two slices: the **enveda-180 validation slice** (test-like acquisition and mass window; the primary slice) and the **mixed validation set** (2,000 spectra from all sources), both against a **230k-molecule candidate pool** (true molecules + 200k decoys) because a 30k pool badly flatters the score.

## Results at a glance
MRR@25 on the 230k-molecule benchmark pool. "Random" = candidates in the mass window ranked in random order (it measures what the mass filter alone achieves; `model − random` is what the model adds).

| Version | Single change | enveda-180 val | mixed val |
|---|---|---|---|
| v0.1 | Ridge on 1 Da bins, ±10 ppm window | 0.1705 | 0.4125 |
| v0.1b | Mass window centred on per-adduct offset | 0.2611 | 0.4125\* |
| v0.2a | + neutral-loss feature block | 0.2767 | 0.4469 |
| v0.2b | 0.1 Da bins (sparse) | 0.3214 | 0.4839 |
| v0.2c | Mass-defect-aware sub-bins (0.02 Da) | 0.4084 | 0.5425 |
| v0.3 | Ridge → MLP at 62k spectra (**not adopted**, negative result) | 0.2832 | 0.4542 |
| **v0.3b** | **10× more data (633k spectra) + deep residual MLP** | **0.6706** | **0.6557** |
| (reference) | random order in the mass window | 0.1835 | 0.3111 |

\*the mixed slice uses the original uncentred 10 ppm window. Pooling a molecule's spectra into one query adds a further +0.03 to +0.06 MRR (deep MLP, enveda-180 molecules: 0.6695 → 0.7314).

Each row changes one thing; the full record with confidence intervals, checks and caveats is in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md). Per-experiment tables are in [`results/`](results/).

**Leaderboard:** 0.076 (v0.3b deep MLP, train-only library). This is the number that matters for the competition and it is far below the local benchmark for the reason given above.

---

## The v0.1 baseline in detail
The first end-to-end pipeline: 1 Da bins + Ridge → Morgan-1024 fingerprints → rank candidates in a 10 ppm window. Code: [`notebooks/enveda-casmi26-baseline.ipynb`](notebooks/enveda-casmi26-baseline.ipynb) (cells 0–49), condensed in [`reference/v0_1_pipeline.py`](reference/v0_1_pipeline.py). Working set: 76,248 spectra, 29,964 molecules (≤5,000 molecules per source, ≤3 spectra each); split 61,701 train / 14,547 validation rows.

![Training data overview](reports/v01_1_data_overview.png)
*Training data after cleaning (200k sample): peaks per spectrum, neutral mass, precursor error, adduct, instrument, source, collision energy, spectra per molecule.*

| Method | MRR@25 |
|---|---|
| A. Frequency floor (always the 25 most common training molecules) | 0.0000 |
| B. Cosine spectral-library match (train-only library) | 0.0000 |
| C. Predict fingerprint, rank train-only library | 0.0000 |
| D. C + 10 ppm mass filter (train-only library) | 0.0000 |
| E1. Oracle library (val molecules added, ~30k pool), no mass filter | 0.0050 (≈40× chance) |
| E2. Oracle library + 10 ppm filter | 0.6752 |
| E3. Oracle + filter, **random** order | 0.5844 (model adds only ≈0.09) |

**What this tells us.** (i) Any method restricted to a train-only library scores exactly 0 on unseen molecules — the candidate pool is the biggest lever. (ii) Inside a 10 ppm window the pool is small (median 3 candidates at 30k), so even random order scores 0.58; E2 is misleadingly high. (iii) 1.65% of true molecules fall outside the window (mostly riken/`[M+Na]+`).

**Scaling benchmark.** Adding up to 200k real decoy molecules to the pool (true molecule always present, 10 ppm) is the honest test, and the one used for all later versions:

![Scaling benchmark](reports/v01_3_scaling_benchmark.png)

| Pool | Cand./query (mean) | Model | Random | Model − random |
|---|---|---|---|---|
| 29,964 | 4.7 | 0.6759 | 0.5878 | 0.088 |
| 39,964 | 6.4 | 0.6275 | 0.5375 | 0.090 |
| 79,964 | 13.7 | 0.5265 | 0.4327 | 0.094 |
| 129,964 | 22.7 | 0.4638 | 0.3665 | 0.097 |
| 229,964 | 40.6 | 0.4021 | 0.3042 | 0.098 |

Source: [`results/v0.1_scaling_benchmark.csv`](results/v0.1_scaling_benchmark.csv). The v0.1 evaluation sample was not seeded, so re-runs differ by ~0.01 (the notebook's inline plot and `docs/EXPERIMENT_LOG.md` quote 0.396–0.423 at 230k); the table and figure here use the saved CSV. The model's advantage over random is flat (~0.09): the Ridge ranker is weak, and at realistic pool sizes mass alone leaves many isomers.

---

## Improvements after v0.1
Every step was judged on the 230k pool with both slices. Paired molecule-clustered bootstrap CIs are in [`results/`](results/) (`*_bootstrap.csv`).

### Understand the data (EDA)
Figures: [`reports/p2_*`](reports/). Findings that shaped the model:
- Spectra of one molecule differ strongly by collision energy (median cosine 20 vs 60 eV = 0.33; chance 0.03), so a molecule should be scored from *all* its spectra together.
- Enveda-180 matches the target acquisition (timsTOF, 20/40/60 eV), but its chemistry is combinatorial and N-rich (100% contain N, 0% acyclic); natural-product sources look different.
- 21% of test-like spectra exceed the v0.1 500-peak cap, so dense spectra should be capped to top-K peaks rather than dropped.

![Same molecule, different measurements](reports/p2_1a_same_molecule_examples.png)
*One molecule measured in different libraries/instruments/collision energies (left) and one spectrum at 20 → 40 → 60 eV (right).*

![np-examples vs rest](reports/p2_4_np_examples_vs_rest.png)
*Enveda natural-product examples vs enveda-180 vs the test-like file: mass range, collision-energy scheme, overlap with other sources, peak counts.*

Other EDA figures: [spectrum agreement](reports/p2_1b_spectrum_agreement.png), [distributions by source](reports/p2_2_distributions_by_source.png), [adduct/polarity/instrument mix](reports/p2_3_adduct_polarity_instrument_mix.png), [molecule properties](reports/p2_5_molecule_properties.png) ([table](reports/p2_5_molecule_properties.csv)).

### v0.1b — Mass-window policy (P3)
`precursor_error_ppm` in the data is unsigned and hides a systematic instrument offset. Recomputing signed errors from the molecular formula gives a per-adduct offset (e.g. `[M+H]+` +2.22 ppm, σ 0.87), so the window is centred on it ([`results/mass_policy_v1.csv`](results/mass_policy_v1.csv), [`src/mass_window.py`](src/mass_window.py)).

![Mass-window policies](reports/p3_4_benchmark_policies.png)
*Test-like slice: MRR, median candidates and miss rate versus window half-width. The star is the adopted policy.*

Test-like slice: 0.1705 → **0.2611** (median candidates 68 → 33, true molecule outside window 0.67%). Most of the gain is fewer candidates, not a better model. Supporting figures: [mass bias](reports/p3_1_mass_bias.png), [miss vs tolerance](reports/p3_2_miss_vs_tolerance.png), [candidates vs tolerance](reports/p3_3_candidates_vs_tolerance.png).

### v0.2a–c — Better spectrum features (P4)
| Step | Idea | enveda-180 | mixed |
|---|---|---|---|
| v0.2a | Add a neutral-loss block (precursor m/z − fragment m/z); shared losses (H₂O, CO…) generalise across molecules | 0.2767 (+0.0156) | 0.4469 |
| v0.2b | 0.1 Da bins instead of 1 Da (keeps part of the mass defect = elemental composition) | 0.3214 (+0.0447) | 0.4839 |
| v0.2c | Spend columns where intense fragments live: 15 × 0.02 Da sub-bins on defect [−0.05, +0.25) per Da | **0.4084** (+0.0870) | **0.5425** |

![Binning schemes](reports/p4c_1_schemes.png)
*The three binning schemes on the mass-defect axis.*

![v0.2c results and checks](reports/p4c_2_results_and_checks.png)
*v0.2c: binning ladder, Ridge-α robustness, gain by source, and the memorisation check (gain by similarity of the validation molecule to its nearest training molecule).*

The gain is partly tied to similar molecules being in train (enveda-180 +0.16 at Tanimoto ≥ 0.5 vs +0.084 at 0.3–0.4; non-enveda-180 +0.039 below 0.3), so natural-product test molecules should see the smaller end. Further figures: [neutral-loss intuition](reports/p4a_1_neutral_loss_intuition.png), [P4a results](reports/p4a_2_results.png), [resolution intuition](reports/p4b_1_resolution_intuition.png), [P4b results](reports/p4b_2_results.png).

### v0.3 — Ridge → MLP, a negative result (P5)
An MLP (BCE on fingerprint bits, scaffold-grouped early stopping) lost to Ridge in all four configurations at 62k training spectra (enveda-180 0.2832 vs 0.4084). A Ridge/MLP blend and a Bernoulli log-likelihood ranking rule did not rescue it.

![P5 results](reports/p5_results.png)

### v0.3b — Data scale and depth (P5b)
The v0.3 verdict was a small-data artefact. With nested training tiers up to 633,198 spectra (every cleaned molecule outside validation, ≤3 spectra each) the MLP overtakes Ridge between 140k and 633k spectra, and a deep residual MLP (Linear 1024 → 4 pre-norm residual blocks → 1,024 sigmoid outputs) reaches **0.6706 / 0.6557**, +0.1005 / +0.0685 over Ridge on the same data.

![Learning curves](reports/p5b_scale.png)
*Learning curves (Ridge vs MLP vs deep MLP), bit-level AUC, MLP-minus-Ridge by tier and by similarity to the large training set.*

![Deep MLP by similarity and source](reports/p5b_deep_novelty.png)
*Deep MLP at tier D: gain by similarity bin and by source.*

| Training tier | spectra | Ridge (enveda-180 / mixed) | MLP 1024×1024 | Deep residual |
|---|---|---|---|---|
| A (= everything before v0.3b) | 61,701 | 0.4084 / 0.5425 | 0.2832 / 0.4542 | 0.2973 / 0.4735 |
| B | 139,679 | 0.4842 / 0.5780 | 0.3903 / 0.5443 | — |
| D | 633,198 | 0.5701 / 0.5871 | 0.5951 / 0.6062 | **0.6706 / 0.6557** |

Tables: [`results/scale_scores.csv`](results/scale_scores.csv), [`scale_bootstrap.csv`](results/scale_bootstrap.csv), [`scale_novelty.csv`](results/scale_novelty.csv), [`scale_auc.csv`](results/scale_auc.csv).

### Candidate-pool cost (P7a) and pooling check
Adding 432k COCONUT structures as decoys (true molecule still in the pool) lowers the deep model's score by about 0.11 on natural-product-like validation sources (median candidates grow 3–5×), while `model − random` stays flat or rises: the cost is more same-mass isomers, not a worse model. Tables: [`results/p7a_pool_cost.csv`](results/p7a_pool_cost.csv), [`p7a_pool_cost_by_source.csv`](results/p7a_pool_cost_by_source.csv), [`pooling_check.csv`](results/pooling_check.csv).

---

## Limitations
- **Candidate pool.** With a train-only library the Kaggle score was 0.076. A rough overlap estimate (0.076 ÷ ~0.6 molecule-level MRR when the truth is in the library) suggests only ~12–14% of hidden-test molecules are reachable. This is an inference, not a measurement.
- **Domain gap.** Local validation is dominated by lab-library chemistry. At tier D every enveda-180 validation molecule has a training neighbour at Tanimoto ≥ 0.4, so that slice partly measures near-neighbour recognition. For genuinely novel molecules (< 0.3) the gain is unconfirmed (n = 97, CI includes 0). Expect natural-product test molecules to behave like the low-similarity bins.
- **The benchmark pool's decoys** are mostly training molecules at tier D (no leak of the validation truth, but unlike the test situation).
- **One seed per run**; CIs cover validation-query sampling, not training randomness. MLP hyper-parameters were tuned at tier A and early-stopped on BCE, not on ranking.
- **The visible `test.parquet` is a placeholder** drawn from training spectra (all 1,213 spectra have exact twins in train); statements derived from it (mass range, adduct mix) describe enveda-180 acquisition, not the hidden test.
- v0.1 benchmark numbers are not exactly reproducible (unseeded evaluation sample); later versions are seeded.
- The Kaggle notebooks were run end-to-end in DEBUG mode locally; the full-size Ridge notebook has not been run on Kaggle (a full-size score of ~0.05 is an estimate, not a measurement).

## Repository layout
```
README.md
requirements.txt
notebooks/enveda-casmi26-baseline.ipynb   v0.1 baseline + EDA/P3/P4/P5 cells (outputs kept for the v0.1 part)
reports/                                  all figures (p2_* EDA, p3_* mass window, p4* features, p5* models, v01_* baseline)
results/                                  per-experiment tables (scores, bootstrap CIs, novelty/source breakdowns, training histories)
docs/EXPERIMENT_LOG.md                    one entry per version: hypothesis, single change, numbers, caveats, decision
src/                                      reusable modules: pipeline, benchmark, mass window, MLP, cache builders
work/                                     experiment scripts (p3_ … p7a_) and the Kaggle-notebook generator
kaggle/                                   Kaggle submission notebooks (Ridge baseline, deep MLP) + run instructions
reference/v0_1_pipeline.py                condensed v0.1 code (reference only, not directly runnable)
```
The Kaggle notebooks are generated from `work/kaggle_cells.py`; edit that, then run `python work/build_kaggle_notebook.py`.

## Reproducing
Data is not included. Download the competition files from the Kaggle page into `data/` (`train.parquet`, `test.parquet`, `sample_submission.csv`).

```bash
pip install -r requirements.txt            # RDKit must be 2026.3.3
python src/build_canon_cache.py            # once: canonical keys + scaffolds (~2 min on 8 cores)
python work/p3_benchmark.py                # v0.1b mass-window benchmark
python work/p4c_defect_bins.py             # v0.2c features + Ridge (needs ~10 GB RAM; run each large fit in its own process)
python work/p5b_build_data.py              # v0.3b: build tiers (~15 min)
python work/p5b_fit.py deep D              # deep residual MLP at tier D (~50 min on Apple MPS)
python work/p5b_score.py                   # score all tiers/models
```
Notes: `src/v1_pipeline.py` still loads the v0.1 decoy files (`dec_keys.npy`, `dec_fp_n.npy`, `dec_mass.npy`) from the working directory; they are produced by the v0.1 notebook (they are not included here, 830 MB). The Ridge solver builds a ~6–8 GB Gram matrix and a 16 GB machine cannot run it alongside GPU training; scripts expect the layout above (`data/`, `artifacts/`). To submit, follow [`kaggle/README.md`](kaggle/README.md); internet is disabled in the competition notebooks, so any external candidate set must be attached as a Kaggle dataset.

## Data, rules and licences
- Competition data is not redistributed here. Figures and tables are derived from it; the competition data are reported to be CC BY-NC 4.0, so keep the project non-commercial and credit the sources.
- The optional COCONUT candidate cache is built from the Kaggle dataset `prvsiyan/coconut-casmi26-candidates` (CC BY-NC-SA 4.0); credit the author if you use it.
- **Check the competition rules on code sharing before making this repository public while the competition is running** (final submission 2026-12-14). Rules wording on external data and notebook requirements was confirmed (public external data and pretrained models are allowed; submissions only via offline Notebooks, ≤ 9 h), but the code-sharing clause has not been read from the Rules page by the author.
- No software licence has been chosen yet.

## Roadmap
1. External candidate pool: COCONUT (cache built), then a mass-sorted PubChem channel, with a popularity prior that does not leak train/COCONUT origin.
2. Learned reranker over the top candidates (Ridge cosine + MLP probability + signed mass error).
3. Similarity-controlled validation (held-out molecule clusters at Tanimoto < 0.4) and a natural-product hold-out before trusting absolute numbers.
4. Re-tune the deep MLP at scale (MRR-aligned early stopping, multiple seeds, all 1.73M spectra).
