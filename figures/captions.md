# Figure and table provenance for the SPMB 2026 submission

The submitted paper (30 June 2026) contains five figures and two tables.
Every number traces to a CSV in `data/` and a script in `src/`. Two data
runs are involved and are listed explicitly so nothing is over-claimed:

| Run | Files | Cells | Used for |
| --- | --- | --- | --- |
| Three-axis run (sampling rate, gain, power line) | `data/sweep_latest.csv`, `data/per_recording_predictions_latest.csv` (7 728 rows) | 28 | Figures 2, 3, 4, 5; Pinsker validation (Sec. IV-C, 18 non-identity arms); failure taxonomy (Sec. IV-D, Table 2) |
| Five-axis run (adds electrode montage and broadband SNR, plus the 60 uV line-noise cell) | `data/sweep_tuab_final.csv` (46 cells), `data/per_recording_predictions_tuab_final.csv` (12 696 rows) | 46 | Table 1 and the per-axis text of Sec. IV-B |

The five-axis run is a superset of the three-axis run: the 28 shared cells
are identical to the decimal. Figures 2-4 were rendered before the two
additional axes were swept and therefore show three axes; the text and
Table 1 report all five.

No clinical recordings are used anywhere in the paper. An earlier draft
contained a small legacy-file ingestion demonstration on clinic data; it
was removed before submission and nothing in this repository depends on
it.

---

**Fig. 1 (fig_architecture.pdf, `src/paper_figures.py`).** ESF inference
pipeline. A native recording is canonicalized into ESF (19-channel 10-20 at
250 Hz, uV-calibrated, robust z-scored, average-referenced, notched) before
encoding by the frozen EEGPT backbone; a TUAB-trained linear probe emits
per-window logits that are Platt-calibrated. The five acquisition-shift
axes inject upstream of canonicalization so naive and canonicalized arms
share every downstream weight.

**Fig. 2 (fig_axes.pdf, `src/paper_figures.py`).** Per-axis
acquisition-shift behaviour on TUAB (n = 276). Dashed line at AUROC 0.908
is the clean reference. ESF canonicalization closes the recoverable axes
(line noise, gain, in-band sampling, bipolar18 montage) and is provably
unable to close the irrecoverable ones (sub-Nyquist sampling, broadband
Gaussian noise). Error bars are 95 % paired-bootstrap CIs. *Rendered from
the three-axis run; the montage and broadband axes appear in Table 1.*

**Fig. 3 (fig_axis_heatmap.pdf, `src/paper_figures.py`).** Delta-AUROC
grid (canon minus clean reference) across (axis, severity). Recoverable
cells (Prop. 1) and irrecoverable cells (Prop. 2) partition before
inspecting the data; cell sign and magnitude match the prediction.
Asterisks mark Benjamini-Hochberg FDR significance at q = 0.05.

**Fig. 4 (fig_failure_taxonomy.pdf, `src/failure_taxonomy.py`).**
Per-cell failure-category composition across the canonicalized arm,
n = 276. Catastrophic shift grows with departure from 256 Hz and gain 1.0x;
the notch keeps the line-noise axis near baseline. The near-absence of
calibration collapse shows that under shift the probe commits with high
confidence, often in the wrong direction.

**Fig. 5 (fig_reliability.pdf, `src/paper_figures.py`).** Reliability
diagram for the Platt-scaled probe on the clean ESF eval split (n = 276).
ECE 0.125 (10 equal-width bins); per-bin Wilson 95 % binomial intervals.

**Table 1 (`data/sweep_tuab_final.csv`).** Five-axis ESF sweep on TUAB
(n = 276): AUROC, balanced accuracy and ECE for the naive and canonicalized
arms at every severity. FDR asterisks from `src/run_all_advanced_stats.py`.

**Table 2 (`data/per_recording_failure_profile.csv`, produced by
`src/failure_taxonomy.py`).** Failure-category prevalence stratified by
ground-truth label: robust-correct 97, shift-recovered 3, calibration
collapse 0, axis-specific 8, shift-induced flip 1, catastrophic shift 112,
robust-wrong 55.

**Pinsker validation (Sec. IV-C; `src/validation_compute.py` writes
`data/theory/validation_kl_pinsker.csv` and `validation_summary.json`).**
Over the 18 non-identity arms of the three-axis run the bound holds on
18/18 with C_max = 0.467 and median implied C = 0.10.

---

## Extended figures (generated, not in the submitted paper)

`src/paper_figures.py` also renders six figures from the three-axis run
that were cut for space: `fig_coverage.pdf`, `fig_roc_pr_curves.pdf`,
`fig_calibration_decomposition.pdf`, `fig_effect_size_forest.pdf`,
`fig_selective_risk_curve.pdf`, `fig_pipeline_flow.pdf`. They are kept
because the advanced-statistics block in `data/advanced_stats_summary.md`
refers to them.
