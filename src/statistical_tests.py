"""
Statistical tests on the SPMB perturbation sweep (paper §Results: per-axis,
statistical-rigor block).

Consumes the per-recording predictions CSV emitted by
``apps.training.spmb_acquisition_shift.eval_loop`` (one row per
(rec_id, axis, severity, arm)) and produces a per-(axis, severity) CSV of
statistical tests comparing the naive and canonicalized arms head-to-head.

For each (axis, severity), with the SAME recordings appearing in both
arms (matched pairing), we compute:

  * AUROC for the naive arm and the canonicalized arm (paired across rec_id)
  * Paired-bootstrap 95% CI on Δ = AUROC_canon - AUROC_naive (n_resamples=1000)
  * DeLong's test p-value for the equality of two correlated ROC curves
    (Sun & Xu 2014 fast algorithm; pure numpy implementation, no extra
    Python dependency required)
  * Cohen's d on per-recording correctness (naive_correct vs canon_correct).
    Computed as paired Cohen's d_z = mean(d) / sd(d), where d = canon - naive
    on the per-recording 0/1 correctness vector. Reported as 0.0 when the
    paired difference has zero variance (no disagreements between arms).

Output CSV columns (one row per (axis, severity)):
    axis, severity, n_recordings, auroc_naive, auroc_canon, auroc_delta,
    delta_ci_lo, delta_ci_hi, p_delong, cohens_d

DeLong implementation notes (Sun & Xu 2014, IEEE SP Letters, "Fast
Implementation of DeLong's Algorithm for Comparing the Areas Under
Correlated Receiver Operating Characteristic Curves"):

  Let m = #positives, n = #negatives. Let X_k, Y_k be the scores under
  classifier k for positive/negative samples. Then for each classifier k:
    V10_i^k = (1/n) * sum_j 1{X_i^k > Y_j^k} + 0.5 * 1{X_i^k == Y_j^k}
    V01_j^k = (1/m) * sum_i 1{X_i^k > Y_j^k} + 0.5 * 1{X_i^k == Y_j^k}
  The empirical AUC is theta_k = mean(V10_i^k) = mean(V01_j^k).
  The 2x2 covariance estimate of (theta_1, theta_2) under DeLong's
  structural framework uses S10 = cov(V10), S01 = cov(V01) and:
    S = (1/m) * S10 + (1/n) * S01
  Test statistic: z = (theta_1 - theta_2) / sqrt(c^T S c) with c=(1,-1)^T,
  two-sided p-value = 2 * (1 - Phi(|z|)). When variance ~ 0 we return
  p = 1.0 (no detectable difference) instead of nan.

Usage:

    PYTHONPATH=. python3 -m apps.training.spmb_acquisition_shift.statistical_tests \\
      --per-rec-csv ~/data/spmb_sweep/per_recording_predictions.csv \\
      --output-csv ~/data/spmb_sweep/statistical_tests.csv \\
      --n-resamples 1000 \\
      --seed 7
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core AUROC + DeLong primitives (pure numpy)
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    """Compute mid-ranks of x (ties get the average rank).

    Replicates scipy.stats.rankdata(x, method='average') in pure numpy so
    the module has zero scipy dependency for the DeLong path.
    """
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[order[j + 1]] == x[order[i]]:
            j += 1
        # average rank (1-indexed) for the tie block [i..j]
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def auroc_paired(labels: np.ndarray, scores: np.ndarray) -> float:
    """Empirical AUROC via the Mann-Whitney U identity.

    AUROC = (R_pos - m*(m+1)/2) / (m*n) where R_pos = sum of ranks of the
    positive class in the combined sorted list (mid-ranks for ties).
    Matches sklearn.metrics.roc_auc_score on ranked-data inputs.
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    m = int(pos.sum())
    n = int((~pos).sum())
    if m == 0 or n == 0:
        return float("nan")
    ranks = _midrank(scores)
    r_pos = float(ranks[pos].sum())
    return (r_pos - m * (m + 1) / 2.0) / (m * n)


def _delong_structural_components(
    labels: np.ndarray,
    scores_arms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (theta, V10, V01) for K classifiers on the same labelled set.

    Args:
      labels: (N,) int {0,1}
      scores_arms: (K, N) float -- one row of scores per classifier/arm

    Returns:
      theta: (K,) empirical AUROCs
      V10:   (K, m) Mann-Whitney structural components on positives
      V01:   (K, n) Mann-Whitney structural components on negatives
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores_arms = np.atleast_2d(np.asarray(scores_arms, dtype=np.float64))
    pos_mask = labels == 1
    m = int(pos_mask.sum())
    n = int((~pos_mask).sum())
    if m == 0 or n == 0:
        raise ValueError("Need at least one positive and one negative example")
    K = scores_arms.shape[0]
    theta = np.empty(K, dtype=np.float64)
    V10 = np.empty((K, m), dtype=np.float64)
    V01 = np.empty((K, n), dtype=np.float64)
    for k in range(K):
        x = scores_arms[k, pos_mask]   # (m,)
        y = scores_arms[k, ~pos_mask]  # (n,)
        # mid-rank within positives, within negatives, within combined
        tx = _midrank(x)          # (m,) ranks among positives only
        ty = _midrank(y)          # (n,) ranks among negatives only
        tz = _midrank(np.concatenate([x, y]))  # (m+n,)
        # Structural components (Sun & Xu, eq. (2.1))
        V10[k, :] = (tz[:m] - tx) / float(n)
        V01[k, :] = 1.0 - (tz[m:] - ty) / float(m)
        theta[k] = float(V10[k, :].mean())
    return theta, V10, V01


def delong_p_value(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[float, float, float, float]:
    """Two-sided DeLong p-value for AUROC_a vs AUROC_b on the same labels.

    Returns (auroc_a, auroc_b, p_value, z_statistic).
    p_value falls back to 1.0 when the estimated variance is non-positive
    (e.g., identical score arrays).
    """
    arms = np.vstack([np.asarray(scores_a, dtype=np.float64),
                      np.asarray(scores_b, dtype=np.float64)])
    theta, V10, V01 = _delong_structural_components(np.asarray(labels), arms)
    m = V10.shape[1]
    n = V01.shape[1]
    # 2x2 covariance under DeLong (ddof=1 for unbiased estimate)
    S10 = np.cov(V10, ddof=1) if m > 1 else np.zeros((2, 2))
    S01 = np.cov(V01, ddof=1) if n > 1 else np.zeros((2, 2))
    # cov() of a 2-row matrix returns shape (2,2)
    S = S10 / float(m) + S01 / float(n)
    c = np.array([1.0, -1.0])
    var_diff = float(c @ S @ c)
    diff = float(theta[0] - theta[1])
    if not np.isfinite(var_diff) or var_diff <= 0.0:
        # Identical or degenerate inputs → no detectable difference.
        return (float(theta[0]), float(theta[1]), 1.0, 0.0)
    z = diff / np.sqrt(var_diff)
    # Two-sided normal tail: 2 * (1 - Phi(|z|)). Use math.erfc for pure
    # numpy/python; equivalent to 2*scipy.stats.norm.sf(|z|).
    from math import erfc, sqrt
    p = float(erfc(abs(z) / sqrt(2.0)))
    # Clamp to [0, 1] (numerical safety)
    p = max(0.0, min(1.0, p))
    return (float(theta[0]), float(theta[1]), p, float(z))


# ---------------------------------------------------------------------------
# Paired bootstrap on AUROC delta
# ---------------------------------------------------------------------------
def paired_bootstrap_auroc_delta_ci(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired bootstrap percentile CI on Δ = AUROC(scores_b) - AUROC(scores_a).

    Resamples RECORDINGS (rows) with replacement; both arms see the same
    rows in each resample so the pairing is preserved (which is the whole
    point of the paired bootstrap when comparing two scorers on the same
    set). Returns (lo, hi) of the percentile CI at the given level.
    Drops resamples that have <2 classes (AUROC undefined).
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    N = labels.shape[0]
    if N == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    deltas[:] = np.nan
    for i in range(n_resamples):
        idx = rng.integers(0, N, size=N)
        yb = labels[idx]
        if len(np.unique(yb)) < 2:
            continue
        a_b = scores_a[idx]
        b_b = scores_b[idx]
        auc_a = auroc_paired(yb, a_b)
        auc_b = auroc_paired(yb, b_b)
        if not (np.isfinite(auc_a) and np.isfinite(auc_b)):
            continue
        deltas[i] = auc_b - auc_a
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - ci_level
    lo = float(np.percentile(deltas, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(deltas, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Cohen's d on per-recording correctness (paired)
# ---------------------------------------------------------------------------
def paired_cohens_d(x_naive: np.ndarray, x_canon: np.ndarray) -> float:
    """Paired Cohen's d_z = mean(d) / sd(d), with d = canon - naive.

    Inputs should already be aligned by recording (same index → same
    recording). On a 0/1 correctness vector this collapses to:
        sum of (canon-naive disagreements signed by direction) / std.
    Returns 0.0 when sd(d) is exactly zero (the two arms agree on every
    recording → no effect to estimate).
    """
    a = np.asarray(x_naive, dtype=np.float64)
    b = np.asarray(x_canon, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    d = b - a
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(d.mean() / sd)


# ---------------------------------------------------------------------------
# Significance asterisk helper (shared with figures.py)
# ---------------------------------------------------------------------------
def significance_marker(p: float) -> str:
    """Map a p-value to a stars/n.s. string.

    Threshold scheme (shared across the figure + table + tests):
      p < 0.001 -> "***"
      p < 0.01  -> "**"
      p < 0.05  -> "*"
      else      -> "n.s."
      nan       -> "n/a"
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# CSV ingest + per-(axis, severity) pairing
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
            })
    return rows


def pair_rows_by_recording(
    rows: Iterable[dict],
    axis: str,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """For one (axis, severity) cell, return aligned (labels, p_naive, p_canon, rec_ids).

    Only recordings that appear in BOTH arms are included (paired set).
    """
    naive: dict[str, dict] = {}
    canon: dict[str, dict] = {}
    for r in rows:
        if r["axis"] != axis or str(r["severity"]) != str(severity):
            continue
        if r["arm"] == "naive":
            naive[r["rec_id"]] = r
        elif r["arm"] == "canonicalized":
            canon[r["rec_id"]] = r
    shared = sorted(set(naive.keys()) & set(canon.keys()))
    if not shared:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                [])
    labels = np.asarray([naive[r]["label"] for r in shared], dtype=np.int64)
    canon_labels = np.asarray([canon[r]["label"] for r in shared], dtype=np.int64)
    if not np.array_equal(labels, canon_labels):
        # Should never happen — same recording must have the same label
        # under both arms. Log and trust the naive-arm copy.
        logger.warning(
            f"label mismatch between arms in axis={axis} severity={severity}; "
            f"using naive-arm labels"
        )
    p_naive = np.asarray([naive[r]["prob"] for r in shared], dtype=np.float64)
    p_canon = np.asarray([canon[r]["prob"] for r in shared], dtype=np.float64)
    return labels, p_naive, p_canon, shared


def compute_statistical_tests(
    rows: list[dict],
    n_resamples: int = 1000,
    seed: int = 7,
) -> list[dict]:
    """Run all stats per (axis, severity), return list of dict rows."""
    # Discover (axis, severity) cells from the per-rec CSV
    cells = sorted({(r["axis"], str(r["severity"])) for r in rows})
    out: list[dict] = []
    for axis, severity in cells:
        labels, p_naive, p_canon, rec_ids = pair_rows_by_recording(
            rows, axis, severity
        )
        n = len(rec_ids)
        if n == 0:
            logger.warning(f"  [{axis}/{severity}] no paired recordings; skipping row")
            continue
        if len(np.unique(labels)) < 2:
            # Single-class set — AUROCs are undefined, but we still emit
            # a row with NaNs so downstream code knows the cell existed.
            row = {
                "axis": axis,
                "severity": severity,
                "n_recordings": n,
                "auroc_naive": float("nan"),
                "auroc_canon": float("nan"),
                "auroc_delta": float("nan"),
                "delta_ci_lo": float("nan"),
                "delta_ci_hi": float("nan"),
                "p_delong": float("nan"),
                "cohens_d": float("nan"),
            }
            out.append(row)
            continue
        auc_n, auc_c, p_dl, _z = delong_p_value(labels, p_naive, p_canon)
        delta = auc_c - auc_n
        ci_lo, ci_hi = paired_bootstrap_auroc_delta_ci(
            labels, p_naive, p_canon,
            n_resamples=n_resamples, seed=seed,
        )
        # Per-recording correctness: 1 if argmax prob matches label, else 0.
        yhat_n = (p_naive >= 0.5).astype(np.int64)
        yhat_c = (p_canon >= 0.5).astype(np.int64)
        correct_n = (yhat_n == labels).astype(np.float64)
        correct_c = (yhat_c == labels).astype(np.float64)
        d = paired_cohens_d(correct_n, correct_c)
        row = {
            "axis": axis,
            "severity": severity,
            "n_recordings": n,
            "auroc_naive": float(auc_n),
            "auroc_canon": float(auc_c),
            "auroc_delta": float(delta),
            "delta_ci_lo": float(ci_lo),
            "delta_ci_hi": float(ci_hi),
            "p_delong": float(p_dl),
            "cohens_d": float(d),
        }
        out.append(row)
        logger.info(
            f"  [{axis}/{severity}] n={n} "
            f"AUROC naive={auc_n:.4f} canon={auc_c:.4f} Δ={delta:+.4f} "
            f"95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}] "
            f"p_DeLong={p_dl:.4g} {significance_marker(p_dl)} "
            f"d={d:+.3f}"
        )
    return out


def write_statistical_tests_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "axis", "severity", "n_recordings",
        "auroc_naive", "auroc_canon", "auroc_delta",
        "delta_ci_lo", "delta_ci_hi", "p_delong", "cohens_d",
    ]
    with open(out_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(fieldnames)
        for r in rows:
            def _fmt(v) -> str:
                if isinstance(v, float):
                    return "nan" if v != v else f"{v:.6f}"
                return str(v)
            wr.writerow([_fmt(r[c]) for c in fieldnames])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--per-rec-csv", type=Path, required=True,
                    help="per_recording_predictions.csv from eval_loop.py")
    ap.add_argument("--output-csv", type=Path, required=True,
                    help="Write statistical-test results here")
    ap.add_argument("--n-resamples", type=int, default=1000,
                    help="Paired-bootstrap resample count for the Δ CI")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rows = load_per_recording_csv(args.per_rec_csv)
    if not rows:
        logger.error("Per-rec CSV is empty — abort")
        return 2
    logger.info(f"Loaded {len(rows)} rows from {args.per_rec_csv}")

    out_rows = compute_statistical_tests(
        rows, n_resamples=args.n_resamples, seed=args.seed
    )
    if not out_rows:
        logger.error("No statistical-test rows produced — abort")
        return 3

    write_statistical_tests_csv(out_rows, args.output_csv)
    logger.info(f"Wrote {len(out_rows)} rows → {args.output_csv}")
    # Headline summary
    n_sig = sum(
        1 for r in out_rows
        if np.isfinite(r["p_delong"]) and r["p_delong"] < 0.05
    )
    logger.info(
        f"Per-(axis,severity) cells with p_DeLong<0.05: {n_sig}/{len(out_rows)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
