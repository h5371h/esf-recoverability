# SPMB 2026 figure captions

All numbers traced to `data/sweep_latest.csv` and
`data/per_recording_predictions_latest.csv` (n = 276 TUAB-eval recordings,
7 728 prediction rows across 28 axis-arms) unless otherwise noted.

> **Note on Figures 5, 11, 13.** The paper's three Magna case-study
> figures (`fig_magna_ood.pdf`, `fig_magna_ood_v2.pdf`,
> `spmb_case_study_fig.pdf`) are intentionally NOT included in this
> public repository because the underlying recordings are clinical EEGs
> from Magna Neurology, Hyderabad and cannot be redistributed. The
> `fig_magna_*` figure-generation functions in `src/paper_figures.py`
> remain present but auto-skip when the source CSV is absent — the
> remaining 10 figures (1–4, 6–10, 12) regenerate identically from the
> shipped TUAB-eval CSVs.

---

**Fig. 1 (fig_architecture.pdf).** End-to-end inference pipeline. The native
recording is canonicalized by ESF (`uV`, 250 Hz, 19-channel 10-20 montage)
before encoding by the frozen EEGPT backbone; a TUAB-trained linear probe
emits per-window logits that are Platt-calibrated and gated by a selective
head. Perturbation axes (sampling rate, gain, power-line noise) are injected
upstream of canonicalization so the two arms (`naive`, `canonicalized`) share
all downstream weights. The selective head abstains when `|logit|` falls below
`tau` or when a Neyman-Pearson test against a clean reference distribution
rejects.

---

**Fig. 2 (fig_axes.pdf).** AUROC (top row) and balanced accuracy (bottom row)
versus shift severity on the TUAB-eval set (n = 276 per condition), broken
out by the three acquisition axes (sampling rate, gain, power-line). Error
bars are 95 % bootstrap CIs (AUROC) or Wilson-approx SE (balanced accuracy).
Power-line interference collapses the naive model to AUROC = 0.612 (95 % CI
[0.543, 0.674]) at 30 uV, whereas the canonicalization arm recovers to AUROC
= 0.909 [0.871, 0.941]. Sampling-rate mismatch and gain mismatch produce
smaller but still significant drops.

---

**Fig. 3 (fig_reliability.pdf).** Reliability diagram for the clean baseline
(sampling rate = 250 Hz, no perturbation, n = 276). Confidence-vs-accuracy
markers and the per-bin gap to the diagonal are plotted; the inset shows the
sample count per probability bin. The model is mildly over-confident in the
high-probability region (Expected Calibration Error = 0.125, Brier = 0.145).
The Murphy decomposition gives reliability = 0.025, resolution = 0.128, and
uncertainty = 0.248.

---

**Fig. 4 (fig_coverage.pdf).** Selective coverage-risk curve on per-recording
predictions across all non-clean axis-severity conditions (n = 3 312
recordings). Adequacy ranking uses `|logit|`; abstention at cov = 0.50 yields
risk = 0.201, falling further as we restrict to the most confident slice.
AURC = 0.252.

---

**Fig. 5 (fig_magna_ood.pdf).** *Not regenerated in the public repo —
requires clinical case-study CSV. See note above.*

---

**Fig. 6 (fig_roc_pr_curves.pdf).** ROC (left) and Precision-Recall (right)
curves with 95 % bootstrap bands (200 resamples). The clean baseline reaches
AUROC = 0.908, AP = 0.909. Under 30 uV simulated power-line interference,
the naive arm collapses to AUROC = 0.500 / AP = 0.294, while the
canonicalization arm holds AUROC = 0.909 / AP = 0.910 — a near-complete
recovery from a chance-level failure.

---

**Fig. 7 (fig_calibration_decomposition.pdf).** Brier decomposition (Murphy
1973) per condition (sampling rate / gain / power-line, naive vs
canonicalized). Bars stack reliability (lower is better) atop
uncertainty - resolution (smaller resolution -> taller bar). Uncertainty
(= base-rate variance, 0.248) is fixed across conditions. Canonicalization
returns Brier to baseline (0.14-0.16) under perturbations that drive the
naive arm to 0.34-0.54.

---

**Fig. 8 (fig_effect_size_forest.pdf).** Forest plot of Hedges' `g` (bias-
corrected) on per-recording `|logit|`, comparing each non-clean condition
to the clean baseline (sampling rate = 250 naive). Whiskers are 95 % CIs;
markers distinguish arm. Power-line interference produces the largest
positive shifts in the naive arm (`g` up to +9.96 at 30 uV), shifts that
the canonicalization arm absorbs (`g` near zero). The pattern is consistent
with hyperinflated, unreliable confidence under noise.

---

**Fig. 9 (fig_axis_heatmap.pdf).** Delta-AUROC grid versus clean baseline
across (axis, severity) rows and arm columns. Cells annotated with effect
size; asterisks mark cells significant after Benjamini-Hochberg FDR control
at alpha = 0.05 over the 26 non-trivial cells (paired bootstrap, 500
resamples). Canonicalization eliminates statistically significant negative
shifts on every power-line condition and recovers most sampling-rate
conditions.

---

**Fig. 10 (fig_selective_risk_curve.pdf).** Full risk-coverage curve under
`|logit|`-ranked abstention, separately for naive and canonicalized arms,
restricted to perturbed inputs (n = 3 312). The canonicalization arm
dominates everywhere: AURC = 0.174 vs 0.367 (E-AURC = 0.120 vs 0.309).
Operating points at coverage 0.50 / 0.75 / 1.00 marked.

---

**Fig. 11 (fig_magna_ood_v2.pdf).** *Not regenerated in the public repo —
requires clinical case-study CSV. See note above.*

---

**Fig. 12 (fig_pipeline_flow.pdf).** Annotated data-flow pipeline showing
where each acquisition perturbation enters (top dashed inset, upstream of
canonicalization) and where the selective head intervenes (bottom dashed
band, downstream of calibration). Block labels report the data shape at
each stage. Complements Fig. 1.

---

**Fig. 13 (spmb_case_study_fig.pdf, legacy).** *Not regenerated in the
public repo — requires clinical case-study CSV. See note above.*
