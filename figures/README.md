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
| Paper item | File | Generator | Data |
| --- | --- | --- | --- |
| Fig. 1 | `fig_architecture.pdf` | `paper_figures.fig_architecture` | none (drawn) |
| Fig. 2 | `fig_axes.pdf` | `paper_figures.fig_axes` | `sweep_latest.csv` (three-axis run) |
| Fig. 3 | `fig_axis_heatmap.pdf` | `paper_figures.fig_axis_heatmap` | `sweep_latest.csv` + `per_recording_predictions_latest.csv` |
| Fig. 4 | `fig_failure_taxonomy.pdf` | `src/failure_taxonomy.py` | `per_recording_predictions_latest.csv` |
| Fig. 5 | `fig_reliability.pdf` | `paper_figures.fig_reliability` | `per_recording_predictions_latest.csv` (clean baseline) |
| Table 1 | (table) | `run_all_advanced_stats.py` for FDR marks | `sweep_tuab_final.csv` (five-axis run) |
| Table 2 | (table) | `src/failure_taxonomy.py` | `per_recording_failure_profile.csv` |
| Extended | `fig_coverage.pdf`, `fig_roc_pr_curves.pdf`, `fig_calibration_decomposition.pdf`, `fig_effect_size_forest.pdf`, `fig_selective_risk_curve.pdf`, `fig_pipeline_flow.pdf` | `paper_figures.py` | three-axis run; not in the submitted paper |

All figures regenerate on a public clone; no clinical data is involved.
See `captions.md` for the full provenance note.

The fonts and DPI in `apply_rcparams()` are IEEE conference-style; the
PDFs render with `pdf.fonttype = 42` so all glyphs are embedded as
TrueType outlines (no font dependency on the renderer).
