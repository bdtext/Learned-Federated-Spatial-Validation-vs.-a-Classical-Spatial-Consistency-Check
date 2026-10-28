# Code: controlled benchmark of classical, learned and federated spatial QC on 2023-2026 station data

Everything was written from scratch in October 2026 (Python 3.14, NumPy 2.4, pandas 3.0, SciPy 1.18, PyTorch 2.14 CPU, python-docx). No code from the earlier manuscript is reused.

## Reproduce
1. Download (requires internet): `python dl_dwd.py select && python dl_dwd.py download`, `python dl_knmi.py`, `python dl_ghcnh.py`.
2. Preprocess: `python prep_dwd.py`, `python prep_knmi.py`, `python prep_ghcnh.py`.
3. Detection benchmark (classical detectors, no-training controls, learned validator with validation-only tuning on seed 0): `python run_detection.py DE 0,1,2,3,4 --tune` (grid via `--grid "32,0.001;32,0.003;64,0.002"`; configurations can be trained in parallel processes with `--only-cfg ch,lr` and are cached as `model_*.pt`), likewise `NL` and `NA`; subtle-magnitude regime: `python run_detection.py DE 0,1 --regime subtle --tag _subtle`; post-hoc check of a cached configuration: `python posthoc_cfg.py DE 64 0.002`.
4. Residency: `python run_residency.py DE 0,1,2` and `python run_residency.py EU 0,1,2 --stride 3` (DE federal states; DE vs NL countries). Arms are saved as they finish (`residency_partial.json`), so an interrupted seed resumes.
5. Robustness: `python run_robustness.py 0 --rounds 5 --stride 4` (all rules) or one rule per process, e.g. `python run_robustness.py 0 --rules median --rounds 5 --stride 4`; finished keys are saved at once and `run_stats.py` merges the per-rule files. The residency diagnostic with full-graph inputs is optional (`run_residency.py ... --fullgraph`).
6. Verification-by-recomputation simulation: `python run_govsim.py`.
7. Audit regeneration, tables and figures: `python reaudit.py`, `python run_stats.py`, `python paper_tables.py`, `python fig_extra.py`. Paper: `python write_paper.py` (composes `paper/paper.md` from the fixed front matter, the Methods and a results narrative whose numbers and decision rules are evaluated from `results/summary/`), then `python build_paper.py ../paper/paper.md ../paper/paper.docx` (plain layout) and `python build_paper_ieee.py ../paper/paper.md ../paper/paper_ieee_access.docx` (IEEE Access layout; author block, biographies and photos from `paper/authors.json` and `paper/photos/`). Package: `python package.py`.
8. Or run everything unattended: `powershell -File launch.ps1` (orchestrator `run_all.py`: thread and memory budgets with a live free-memory check, dependencies, retries, per-job logs in `results/log_*.txt`; `awake.py` keeps Windows from idle-sleeping while it runs). Every job is resumable (completed seeds, arms and attack keys are skipped; trained detection models are cached as `model_*.pt`), so the orchestrator can simply be relaunched after an interruption; jobs already running are recognized and never duplicated. Thread count per process: environment variable `QC_THREADS`.

## Modules (`qc/`)
- `data.py`: network assembly, WMO-style plausibility screen, duplicate audit, splits, harmonic climatology and scales.
- `inject.py`: seeded five-family fault injection with magnitudes in units of the clean sigma_a (tabulated in physical units) and an independent audit that reconstructs each segment from the injected-minus-clean difference.
- `detectors.py`: CLIM-Z, EWMA, local-level Kalman, persistence innovation, correlation-weighted leave-one-out SCT with a regression slope, distance/elevation optimal-interpolation SCT (SCT-OI), run-length and variance pre-tests, robust standardization, two-timescale scoring, validation-selected grids, label-assisted and label-free thresholds, AUPRC.
- `model.py`: leave-self-out spatiotemporal graph validator (dilated causal TCN, graph attention with a hybrid prior, forecast head, leave-self-out spatial head with masked diagonal and embedding-only query), training with early stopping on validation loss, residual extraction. The leave-self-out property is verified by a gradient check (`run_smoke` logs).
- `fl.py`: federated rounds over silos (shared backbone averaged, local parameters kept), aggregation rules (mean, coordinate median, multi-Krum, FLTrust) and attacks (scaled sign-flip, a-little-is-enough, min-max).
- `metrics.py`: point-wise metrics, matched operating point, per-family F1, event-level rates (any-flag and persistence criteria), chance-level events, censored delays, seed-level paired t-intervals, TOST equivalence, paired moving-block bootstrap.

## Design choices that answer the earlier reviews
- Validation covers a full annual cycle; thresholds are reported both label-assisted and label-free.
- The learned validator is tuned on validation only (seed 0), with early stopping and learning curves; its untrained state equals the classical estimate, so the learning curve from epoch 0 shows what training adds.
- The classical comparator exists in a simplified (correlation-weighted) and an operational-style (OI with distance and elevation, plus persistence and variance pre-tests) form.
- The residency experiment contains the 2 x 2 design (pooled vs silo training) x (full vs confined graph) and a local-only arm.
- Robustness uses an admissible multi-Krum configuration (K = 8, f <= 2) and adaptive attacks, not only the sign-flip.
- Every number in the paper is generated from `results/` by `run_stats.py`.
