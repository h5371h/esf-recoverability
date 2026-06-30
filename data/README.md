# `data/` — pre-computed results and the TUAB manifest

Everything in this folder is licence-clean: the TUH alphanumeric
recording identifiers carry no PHI under the NEDC data-use agreement,
and the per-recording prediction CSV exposes only the model's logit /
probability / per-recording label.

## Files

| File | Rows | What it is |
| --- | ---: | --- |
| `sweep_latest.csv` | 28 | (axis, severity, arm) → AUROC / bal-acc / ECE / 95 % bootstrap CI |
| `per_recording_predictions_latest.csv` | 7 728 | (rec_id, axis, severity, arm) → label, logit, prob, var_logit |
| `manifest_tuab_eval.json` | – | 276 TUAB rec_ids + patient/label counts |
| `seeds.json` | – | every RNG seed used (default = 7) |
| `advanced_stats_results.json` | – | FDR, mixed-effects, AURC, conformal, MI, power, … |
| `advanced_stats_summary.md` | – | human-readable digest of the above |

`per_recording_predictions_latest.csv` columns:

| column | type | meaning |
| --- | --- | --- |
| `rec_id` | str | TUAB filename stem `<patient>_s<session>_t<token>` |
| `axis` | str | one of {sampling_rate, calibration, power_line, broadband_noise, montage} |
| `severity` | str | level on the axis grid (see `src/perturbations.py`) |
| `arm` | str | `naive` (raw) or `canonicalized` (post-ESF) |
| `label` | int | 0 = normal, 1 = abnormal (TUAB folder layout) |
| `logit` | float | Head A v1 EEGPT linear-probe logit |
| `prob` | float | sigmoid(logit) |
| `var_logit` | float | per-recording variance across windows |

## How to get the TUH source EDFs

The paper's reproduction is bit-exact only if you run against the same
TUAB v3.0.1 eval split listed in `manifest_tuab_eval.json`.

1. Apply for a NEDC data-use agreement at
   <https://isip.piconepress.com/projects/nedc/html/forms/dat_use.shtml>
   (free, institutional email, ~ 1–2 day turnaround).
2. Once approved you receive an rsync username; fetch only the eval
   split needed by the paper:
   ```bash
   rsync -auxvL "${TUH_USER}@www.isip.piconepress.com:~/data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf/eval/" \
                "$TUH_ROOT/edf/eval/"
   ```
   This costs ~ 8 GB on disk.
3. Cross-check the layout against the manifest — every rec_id should
   resolve to `$TUH_ROOT/edf/eval/{normal,abnormal}/<rec_id>.edf`.

## Why no clinical / Magna data is shipped

The paper's Sec. 5.5 case study uses five Natus-format `.e` files from
Magna Neurology, Hyderabad. These are clinical EEGs and cannot be
redistributed under MIT. The figure-generation code
(`src/paper_figures.py`) auto-skips the two Magna panels when the
backing CSV is absent — every other figure and statistic regenerates
identically from the TUAB CSVs shipped above.
