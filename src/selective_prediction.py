"""
Calibrated abstention — coverage vs risk sweep (paper §Coverage--risk).

The paper's adequacy score is defined as:

    "the empirical percentile of a one-class statistic fit on
     clinician-marked adequate recordings"

In this study we use a deployable proxy that requires no clinician labels:
the WINDOW-LOGIT VARIANCE of Head A v1 on each recording.

Why window-logit variance:
  * In-distribution recordings — the EEGPT backbone + Head A v1 produce
    consistent per-window logits → low variance.
  * Out-of-distribution / perturbed recordings — windows disagree (some
    look ID, some look bizarre) → high variance.
  * It needs no extra training and no external one-class fit. The
    empirical-percentile rule converts the raw variance into a 0..1
    adequacy score by ranking against the CLEAN baseline variance
    distribution (axis="sampling_rate", severity=250, arm="naive" rows in
    the per-recording CSV are the cleanest in-distribution condition we
    actually have on TUAB eval).

The output is a coverage-risk curve:

    threshold, coverage, risk, n_retained, n_total

  * threshold  : abstain when adequacy < threshold
  * coverage   : fraction of recordings retained (not abstained on)
  * risk       : 1 - balanced_accuracy on the retained set
  * n_retained : count retained
  * n_total    : total recordings considered for that threshold

By default the script considers mixed-severity perturbed input: every row
in per_recording_predictions.csv whose (axis, severity) is NOT the clean
baseline. This matches the paper's "calibrated abstention rule under
mixed-severity input" framing.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV ingest
# ---------------------------------------------------------------------------
def load_per_recording_csv(path: Path) -> list[dict]:
    """Load per_recording_predictions.csv into a list of typed dicts."""
    rows: list[dict] = []
    with open(path, "r", newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            rows.append({
                "rec_id":   row["rec_id"],
                "axis":     row["axis"],
                "severity": row["severity"],
                "arm":      row["arm"],
                "label":    int(row["label"]),
                "logit":    float(row["logit"]),
                "prob":     float(row["prob"]),
                "var_logit": float(row["var_logit"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Adequacy score
# ---------------------------------------------------------------------------
def fit_adequacy_baseline(rows: list[dict],
                          clean_axis: str = "sampling_rate",
                          clean_severity: str = "250",
                          clean_arm: str = "naive") -> np.ndarray:
    """Pull the window-logit variances on the cleanest baseline condition.

    These define the in-distribution variance distribution against which
    we compute empirical-percentile adequacy.
    """
    baseline_vars = [
        r["var_logit"] for r in rows
        if r["axis"] == clean_axis
        and str(r["severity"]) == str(clean_severity)
        and r["arm"] == clean_arm
    ]
    if not baseline_vars:
        # Fallback — use ALL canonicalized arm rows from the lowest-severity
        # condition of each axis. Still in-distribution-ish.
        baseline_vars = [
            r["var_logit"] for r in rows if r["arm"] == "canonicalized"
        ]
    if not baseline_vars:
        raise RuntimeError(
            "Cannot fit adequacy baseline — no clean rows found in CSV"
        )
    arr = np.asarray(baseline_vars, dtype=np.float64)
    arr.sort()
    logger.info(
        f"Adequacy baseline n={len(arr)} var_logit "
        f"[min={arr.min():.4f} med={np.median(arr):.4f} max={arr.max():.4f}]"
    )
    return arr


def adequacy_score(var_logit: float, baseline_sorted: np.ndarray) -> float:
    """Empirical-percentile adequacy in [0, 1].

    Adequacy = 1 - (rank of var_logit in baseline / n_baseline)

    A recording whose var_logit is LOWER than all clean baseline
    recordings gets adequacy 1.0 (most adequate). A recording whose
    var_logit is HIGHER than all clean baseline recordings gets adequacy
    0.0 (least adequate, should be abstained on).
    """
    n = len(baseline_sorted)
    if n == 0:
        return 0.0
    # searchsorted returns the insertion index in [0, n].
    rank = int(np.searchsorted(baseline_sorted, var_logit, side="right"))
    return float(1.0 - rank / n)


# ---------------------------------------------------------------------------
# Coverage-risk sweep
# ---------------------------------------------------------------------------
def compute_coverage_risk(
    rows: list[dict],
    baseline_sorted: np.ndarray,
    thresholds: np.ndarray,
    clean_axis: str = "sampling_rate",
    clean_severity: str = "250",
) -> list[dict]:
    """Sweep thresholds, return per-threshold coverage + risk dicts.

    Risk is 1 - balanced_accuracy on the retained set.
    """
    try:
        from sklearn.metrics import balanced_accuracy_score  # type: ignore
    except ImportError:
        # Lightweight fallback for laptops without sklearn (the VM has it).
        # balanced_accuracy = (TPR + TNR) / 2 for binary labels {0, 1}.
        def balanced_accuracy_score(y_true, y_pred):  # type: ignore[no-redef]
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            tp = float(((y_true == 1) & (y_pred == 1)).sum())
            tn = float(((y_true == 0) & (y_pred == 0)).sum())
            p = float((y_true == 1).sum())
            n = float((y_true == 0).sum())
            tpr = tp / p if p > 0 else 0.0
            tnr = tn / n if n > 0 else 0.0
            return (tpr + tnr) / 2.0

    # Mixed-severity perturbed pool: every row that is NOT the clean baseline.
    perturbed = [
        r for r in rows
        if not (
            r["axis"] == clean_axis
            and str(r["severity"]) == str(clean_severity)
            and r["arm"] == "naive"
        )
    ]
    if not perturbed:
        raise RuntimeError("No perturbed rows — coverage-risk sweep impossible")

    adequacies = np.asarray(
        [adequacy_score(r["var_logit"], baseline_sorted) for r in perturbed],
        dtype=np.float64,
    )
    labels = np.asarray([r["label"] for r in perturbed], dtype=np.int64)
    probs = np.asarray([r["prob"] for r in perturbed], dtype=np.float64)
    yhat = (probs >= 0.5).astype(np.int64)

    out: list[dict] = []
    n_total = len(perturbed)
    for thr in thresholds:
        keep_mask = adequacies >= thr
        n_retained = int(keep_mask.sum())
        if n_retained == 0 or len(np.unique(labels[keep_mask])) < 2:
            risk = float("nan")
            coverage = float(n_retained) / float(n_total)
            out.append({
                "threshold": float(thr),
                "coverage":  coverage,
                "risk":      risk,
                "n_retained": n_retained,
                "n_total":   n_total,
            })
            continue
        bal = float(balanced_accuracy_score(labels[keep_mask], yhat[keep_mask]))
        risk = 1.0 - bal
        coverage = float(n_retained) / float(n_total)
        out.append({
            "threshold": float(thr),
            "coverage":  coverage,
            "risk":      risk,
            "n_retained": n_retained,
            "n_total":   n_total,
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--per-rec-csv", type=Path, required=True,
                    help="Output of eval_loop.py (per_recording_predictions.csv)")
    ap.add_argument("--output-csv", type=Path, required=True,
                    help="Write coverage-risk sweep here")
    ap.add_argument("--n-thresholds", type=int, default=99,
                    help="Threshold grid size for linspace(0.01, 0.99, N)")
    ap.add_argument("--clean-axis", type=str, default="sampling_rate")
    ap.add_argument("--clean-severity", type=str, default="250",
                    help="Severity value that marks the clean baseline row")
    ap.add_argument("--clean-arm", type=str, default="naive")
    args = ap.parse_args(argv)

    rows = load_per_recording_csv(args.per_rec_csv)
    if not rows:
        logger.error("Per-rec CSV is empty — abort")
        return 2
    logger.info(f"Loaded {len(rows)} rows from {args.per_rec_csv}")

    baseline = fit_adequacy_baseline(
        rows,
        clean_axis=args.clean_axis,
        clean_severity=args.clean_severity,
        clean_arm=args.clean_arm,
    )
    thresholds = np.linspace(0.01, 0.99, args.n_thresholds)
    curve = compute_coverage_risk(
        rows, baseline, thresholds,
        clean_axis=args.clean_axis,
        clean_severity=args.clean_severity,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["threshold", "coverage", "risk", "n_retained", "n_total"])
        for r in curve:
            wr.writerow([
                f"{r['threshold']:.4f}",
                f"{r['coverage']:.6f}",
                "nan" if r["risk"] != r["risk"] else f"{r['risk']:.6f}",
                r["n_retained"],
                r["n_total"],
            ])
    logger.info(f"Wrote coverage-risk CSV → {args.output_csv}")

    # Headline operating point: risk < 0.10 with max coverage.
    best = None
    for r in curve:
        if r["risk"] == r["risk"] and r["risk"] < 0.10:
            if best is None or r["coverage"] > best["coverage"]:
                best = r
    if best:
        logger.info(
            f"Operating point (risk<0.10): threshold={best['threshold']:.3f}  "
            f"coverage={best['coverage']:.3f}  risk={best['risk']:.4f}  "
            f"n_retained={best['n_retained']}/{best['n_total']}"
        )
    else:
        logger.info("No threshold achieves risk<0.10 in this sweep")
    return 0


if __name__ == "__main__":
    sys.exit(main())
