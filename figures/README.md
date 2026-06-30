# `figures/` — paper PDFs

Every figure here is the exact PDF that appears in the SPMB 2026
submission. The script that produces them is `src/paper_figures.py`,
and the captions live in `figures/captions.md`.

To regenerate the entire set from a fresh checkout:

```bash
python src/paper_figures.py
```

The script reads only `data/sweep_latest.csv` and
`data/per_recording_predictions_latest.csv`. It overrides nothing in
place — each PDF is written to `figures/<name>.pdf`.

## Figure → data → script map

| Figure | Function in `paper_figures.py` | Source data |
| --- | --- | --- |
| Fig. 1 `fig_architecture.pdf` | `fig_architecture` | none (drawn) |
| Fig. 2 `fig_axes.pdf` | `fig_axes` | `sweep_latest.csv` |
| Fig. 3 `fig_reliability.pdf` | `fig_reliability` | `per_recording_predictions_latest.csv` (clean baseline) |
| Fig. 4 `fig_coverage.pdf` | `fig_coverage` | `per_recording_predictions_latest.csv` (non-clean conditions) |
| Fig. 5 `fig_magna_ood.pdf` | `fig_magna_ood` | *clinical, not in repo* |
| Fig. 6 `fig_roc_pr_curves.pdf` | `fig_roc_pr_curves` | `per_recording_predictions_latest.csv` |
| Fig. 7 `fig_calibration_decomposition.pdf` | `fig_calibration_decomposition` | `per_recording_predictions_latest.csv` |
| Fig. 8 `fig_effect_size_forest.pdf` | `fig_effect_size_forest` | `per_recording_predictions_latest.csv` |
| Fig. 9 `fig_axis_heatmap.pdf` | `fig_axis_heatmap` | both CSVs |
| Fig. 10 `fig_selective_risk_curve.pdf` | `fig_selective_risk_curve` | `per_recording_predictions_latest.csv` |
| Fig. 11 `fig_magna_ood_v2.pdf` | `fig_magna_ood_v2` | *clinical, not in repo* |
| Fig. 12 `fig_pipeline_flow.pdf` | `fig_pipeline_flow` | none (drawn) |

Figures 5, 11, and 13 (the legacy Magna composite) are auto-skipped on
a public clone because the underlying clinical CSV is not
redistributable. See `captions.md` for the disclosure.

The fonts and DPI in `apply_rcparams()` are IEEE conference-style; the
PDFs render with `pdf.fonttype = 42` so all glyphs are embedded as
TrueType outlines (no font dependency on the renderer).
