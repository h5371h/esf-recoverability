# Random number generation in the SPMB 2026 sweep

Every script that takes a `--seed` flag defaults to **7**, and every
seed is stored in `data/seeds.json` for machine-readable reference.

## Where randomness enters

| Component | RNG source | Default seed | Stored |
| --- | --- | --- | --- |
| Bootstrap AUROC CIs in `eval_loop.py` | `numpy.random.default_rng(seed)` | 7 | sidecar JSON next to sweep CSV |
| Paired bootstrap on AUROC delta in `statistical_tests.py` | `numpy.random.default_rng(seed)` | 7 | CLI flag |
| Mixed-effects logistic init (`statsmodels.MixedLM`) | `np.random.seed(seed)` | 7 | `seeds.json` |
| Stratified vs naive bootstrap (`advanced_stats.py`) | `numpy.random.default_rng(seed)` | 7 | `seeds.json` |
| Permutation test (10 000 permutations) | `numpy.random.default_rng(seed)` | 7 | `seeds.json` |
| Effect-size bootstrap CIs | `numpy.random.default_rng(seed)` | 7 | `seeds.json` |
| AURC bootstrap CIs | `numpy.random.default_rng(seed)` | 7 | `seeds.json` |
| Split-conformal calibration split | `numpy.random.default_rng(seed)` | 7 | `seeds.json` |
| Mutual-information k-NN resampling | `numpy.random.default_rng(seed)` | 7 | `seeds.json` |
| Seed-sensitivity proxy (resampled splits) | `numpy.random.default_rng(seed + i)` for i in 0..K | 7 | `seeds.json` |

## Deterministic-by-construction (no RNG)

* The 5 perturbation functions in `perturbations.py` are pure
  numpy/scipy and produce the same output bit-for-bit for the same
  signal across runs.
* DeLong's test in `statistical_tests.py` is closed-form on mid-rank
  V-statistics — no Monte Carlo step.
* ECE, Brier score, balanced accuracy, AUROC — all closed-form.
* ESF canonicalization (channel mapping, resampling, notch, common
  average reference, robust z-score) is fully deterministic.

## Patient-stratified split protocol

The TUAB v3.0.1 eval split shipped by NEDC is already patient-stratified
— each patient appears in either train or eval, never both. The paper
runs against the **eval split only** (276 recordings, 253 unique
patients — see `data/manifest_tuab_eval.json`), so no additional
patient stratification is required.

The paired-bootstrap CIs and the mixed-effects logistic model both
include a random intercept on `patient_id` (extracted from the rec_id
prefix). See `src/advanced_stats.py::run_method_3_mixed_effects` for
the GLMM specification.

## Overriding the seed

Every CLI takes a `--seed` flag and accepts any non-negative integer:

```bash
python src/run_all_advanced_stats.py --seed 42
```

After the run, `data/advanced_stats_results.json` will contain a
`_meta.seed` field reflecting the seed actually used.
