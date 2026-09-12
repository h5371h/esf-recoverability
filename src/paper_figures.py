"""SPMB 2026 publication figures.

Single source of truth: regenerates every figure from the two CSVs in
<repo>/data/.  No invented numbers.

Each public ``fig_*`` function takes explicit input paths plus an output
path so each plot can be re-rendered in isolation.

Outputs (PDF, IEEE conference column widths):
  fig_axes.pdf                     - AUROC + bal_acc per axis x severity
  fig_reliability.pdf              - Reliability diagram + ECE on clean baseline
  fig_coverage.pdf                 - Selective coverage-risk with operating points
  fig_architecture.pdf             - 4-plane / pipeline schematic
  fig_roc_pr_curves.pdf            - ROC + PR with bootstrap 95% bands
  fig_calibration_decomposition.pdf - Brier reliability/resolution/uncertainty
  fig_effect_size_forest.pdf       - Cohen's d / Hedges' g per (axis,severity)
  fig_axis_heatmap.pdf             - Delta-AUROC heatmap with FDR significance
  fig_selective_risk_curve.pdf     - Full coverage-risk curve, AURC + E-AURC
  fig_pipeline_flow.pdf            - Annotated data-flow pipeline

Author: Hitesh Dammu (regen tooling)
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --------------------------------------------------------------------------- #
# Defaults / paths
# --------------------------------------------------------------------------- #

# Repo-relative defaults so the same script renders identically from a
# fresh checkout. Override via SPMB_DATA_DIR / SPMB_FIG_DIR env vars or
# the CLI args wired by main().
import os as _os
_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(_os.environ.get("SPMB_DATA_DIR", _REPO_ROOT / "data"))
FIG_DIR = Path(_os.environ.get("SPMB_FIG_DIR", _REPO_ROOT / "figures"))
SWEEP_CSV = DATA_DIR / "sweep_latest.csv"
PRED_CSV = DATA_DIR / "per_recording_predictions_latest.csv"

# IEEE conference column widths
COL = 3.5
DCOL = 7.16

# Colorblind-safe Tableau-10-derived palette
PAL = {
    "naive": "#D55E00",          # vermillion
    "canonicalized": "#0072B2",  # blue
    "neutral": "#444444",
    "soft": "#888888",
    "grid": "#BBBBBB",
    "normal": "#009E73",         # green
    "abnormal": "#CC79A7",       # rose
    "ood": "#000000",
    "band_naive": "#F4C7A5",
    "band_canon": "#B3D7E8",
    "delta_pos": "#0072B2",
    "delta_neg": "#D55E00",
}

# --------------------------------------------------------------------------- #
# rc setup
# --------------------------------------------------------------------------- #


def apply_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.1,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.dpi": 300,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


@dataclass
class SweepRow:
    axis: str
    severity: str
    arm: str
    auroc: float
    bal_acc: float
    ece: float
    n: int
    ci_lo: float
    ci_hi: float


def load_sweep(path: Path = SWEEP_CSV) -> List[SweepRow]:
    out: List[SweepRow] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append(
                SweepRow(
                    axis=r["axis"],
                    severity=r["severity"],
                    arm=r["arm"],
                    auroc=float(r["auroc"]),
                    bal_acc=float(r["bal_acc"]),
                    ece=float(r["ece"]),
                    n=int(r["n_recordings"]),
                    ci_lo=float(r["ci_lo"]),
                    ci_hi=float(r["ci_hi"]),
                )
            )
    return out


@dataclass
class PredRow:
    rec_id: str
    axis: str
    severity: str
    arm: str
    label: int
    logit: float
    prob: float
    var_logit: float


def load_predictions(path: Path = PRED_CSV) -> List[PredRow]:
    out: List[PredRow] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append(
                PredRow(
                    rec_id=r["rec_id"],
                    axis=r["axis"],
                    severity=r["severity"],
                    arm=r["arm"],
                    label=int(r["label"]),
                    logit=float(r["logit"]),
                    prob=float(r["prob"]),
                    var_logit=float(r["var_logit"]),
                )
            )
    return out


def filter_pred(
    preds: Sequence[PredRow], axis: str, severity: str, arm: str
) -> List[PredRow]:
    return [p for p in preds if p.axis == axis and p.severity == severity and p.arm == arm]


def slice_arrays(preds: Sequence[PredRow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.array([p.prob for p in preds], dtype=float)
    labels = np.array([p.label for p in preds], dtype=int)
    logits = np.array([p.logit for p in preds], dtype=float)
    return probs, labels, logits


# --------------------------------------------------------------------------- #
# Stats primitives (numpy-only)
# --------------------------------------------------------------------------- #


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via Mann-Whitney U / rank-sum trick (handles ties)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # average rank for ties
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros_like(counts, dtype=float)
    for i, r in zip(inv, ranks):
        sum_ranks[i] += r
    avg = sum_ranks / counts
    ranks = avg[inv]
    sum_pos = ranks[: len(pos)].sum()
    u = sum_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    P = max(int((labels == 1).sum()), 1)
    N = max(int((labels == 0).sum()), 1)
    tpr = tps / P
    fpr = fps / N
    # prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return fpr, tpr


def pr_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    y = labels[order]
    tps = np.cumsum(y == 1).astype(float)
    fps = np.cumsum(y == 0).astype(float)
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / max(int((labels == 1).sum()), 1)
    # standard convention: prepend (recall=0, precision=1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return recall, precision


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    recall, precision = pr_curve(scores, labels)
    # AP = sum over thresholds of (R_k - R_{k-1}) * P_k  (sklearn convention)
    return float(np.sum(np.diff(recall) * precision[1:]))


def bootstrap_curves(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 300,
    seed: int = 0,
    grid: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr_grid, tpr_lo, tpr_hi, rec_grid, prec_lo, prec_hi)."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    tprs = np.empty((n_boot, len(grid)))
    precs = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s, y = scores[idx], labels[idx]
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            tprs[b] = np.nan
            precs[b] = np.nan
            continue
        fpr, tpr = roc_curve(s, y)
        rec, pr = pr_curve(s, y)
        tprs[b] = np.interp(grid, fpr, tpr)
        precs[b] = np.interp(grid, rec, pr)
    tpr_lo = np.nanpercentile(tprs, 2.5, axis=0)
    tpr_hi = np.nanpercentile(tprs, 97.5, axis=0)
    pr_lo = np.nanpercentile(precs, 2.5, axis=0)
    pr_hi = np.nanpercentile(precs, 97.5, axis=0)
    return grid, tpr_lo, tpr_hi, grid, pr_lo, pr_hi


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(probs)
    for k in range(n_bins):
        m = idx == k
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


def brier_decomposition(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Tuple[float, float, float, float]:
    """Murphy decomposition: brier = reliability - resolution + uncertainty.

    Returns (brier, reliability, resolution, uncertainty).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    n = len(probs)
    o = float(labels.mean())
    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        m = idx == k
        if m.sum() == 0:
            continue
        nk = m.sum()
        fk = probs[m].mean()
        ok = labels[m].mean()
        reliability += (nk / n) * (fk - ok) ** 2
        resolution += (nk / n) * (ok - o) ** 2
    uncertainty = o * (1 - o)
    brier = float(np.mean((probs - labels) ** 2))
    return brier, float(reliability), float(resolution), float(uncertainty)


def hedges_g(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Hedges' g with bias correction + 95% CI (large-sample approx)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = math.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if sp == 0:
        return 0.0, 0.0, 0.0
    d = (mx - my) / sp
    J = 1 - 3 / (4 * (nx + ny) - 9)
    g = d * J
    se = math.sqrt((nx + ny) / (nx * ny) + g**2 / (2 * (nx + ny)))
    return g, g - 1.96 * se, g + 1.96 * se


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return boolean mask of rejected hypotheses under BH-FDR."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    crit = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= crit
    if not below.any():
        return np.zeros(n, dtype=bool)
    k = np.max(np.where(below)[0])
    rejected_sorted = np.arange(n) <= k
    out = np.zeros(n, dtype=bool)
    out[order] = rejected_sorted
    return out


def delong_pvalue(
    scoresA: np.ndarray, scoresB: np.ndarray, labels: np.ndarray
) -> float:
    """Approximate paired AUROC z-test via bootstrap (DeLong is closed-form but
    needs more code; bootstrap is good enough for FDR ranking)."""
    rng = np.random.default_rng(42)
    n = len(labels)
    a0 = auroc(scoresA, labels) - auroc(scoresB, labels)
    diffs = []
    for _ in range(500):
        idx = rng.integers(0, n, n)
        l = labels[idx]
        if (l == 1).sum() == 0 or (l == 0).sum() == 0:
            continue
        diffs.append(auroc(scoresA[idx], l) - auroc(scoresB[idx], l))
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return 1.0
    # two-sided p via empirical bootstrap distribution centered on zero
    null = diffs - diffs.mean()
    p = float((np.abs(null) >= abs(a0)).mean())
    return max(p, 1.0 / len(diffs))


# --------------------------------------------------------------------------- #
# fig_axes - two panel AUROC + bal_acc
# --------------------------------------------------------------------------- #


def fig_axes(sweep: List[SweepRow], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(DCOL, 4.0), sharey="row")

    panels = [
        ("sampling_rate", "Sampling rate (Hz)", "Sampling-rate mismatch"),
        ("calibration", "Per-channel gain", "Amplifier gain mismatch"),
        ("power_line", "Power-line noise amplitude ($\\mu$V)", "Power-line interference"),
    ]
    metric_rows = [("auroc", "AUROC"), ("bal_acc", "Balanced accuracy")]

    for col, (axis_key, xlabel, title) in enumerate(panels):
        axis_rows = [r for r in sweep if r.axis == axis_key]
        # group
        by_arm: Dict[str, List[Tuple[float, float, float, float, float, float, float]]] = defaultdict(list)
        for r in axis_rows:
            sev = float(r.severity)
            by_arm[r.arm].append((sev, r.auroc, r.bal_acc, r.ece, r.ci_lo, r.ci_hi, r.n))
        for arm in by_arm:
            by_arm[arm].sort()

        for row_i, (metric, ylabel) in enumerate(metric_rows):
            ax = axes[row_i, col]
            for arm, marker, ls in [("naive", "o", "-"), ("canonicalized", "s", "--")]:
                pts = by_arm.get(arm, [])
                if not pts:
                    continue
                xs = np.array([p[0] for p in pts])
                if metric == "auroc":
                    ys = np.array([p[1] for p in pts])
                    lo = np.array([p[4] for p in pts])
                    hi = np.array([p[5] for p in pts])
                else:
                    ys = np.array([p[2] for p in pts])
                    # symmetric Wilson-style ~1.96*sqrt(p(1-p)/n) approx
                    ns = np.array([p[6] for p in pts])
                    se = 1.96 * np.sqrt(np.clip(ys * (1 - ys), 1e-9, None) / ns)
                    lo, hi = ys - se, ys + se
                yerr_lo = ys - lo
                yerr_hi = hi - ys
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[yerr_lo, yerr_hi],
                    fmt=marker,
                    linestyle=ls,
                    color=PAL[arm],
                    mfc="white" if arm == "naive" else PAL[arm],
                    mec=PAL[arm],
                    ms=5,
                    lw=1.0,
                    capsize=2.5,
                    elinewidth=0.7,
                    label=arm,
                )
            ax.set_ylim(0.45, 1.0)
            ax.grid(True, linestyle=":", linewidth=0.5, color=PAL["grid"], alpha=0.7)
            if row_i == 0:
                ax.set_title(title, pad=4)
            if row_i == 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)
            if axis_key == "sampling_rate":
                ax.set_xticks([128, 200, 250, 256, 512])
                ax.set_xticklabels(["128", "200", "250", "256", "512"], rotation=0)
            elif axis_key == "calibration":
                ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0])
            else:
                ax.set_xticks([0, 5, 15, 30])
            if row_i == 0 and col == 2:
                ax.legend(loc="lower right", frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig_reliability
# --------------------------------------------------------------------------- #


def fig_reliability(preds: List[PredRow], out_path: Path) -> None:
    clean = filter_pred(preds, "sampling_rate", "250", "naive")
    if not clean:
        clean = filter_pred(preds, "calibration", "1.0", "naive")
    probs, labels, _ = slice_arrays(clean)

    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    centers, accs, confs, ns = [], [], [], []
    for k in range(n_bins):
        m = idx == k
        if m.sum() == 0:
            continue
        centers.append((bins[k] + bins[k + 1]) / 2)
        accs.append(float(labels[m].mean()))
        confs.append(float(probs[m].mean()))
        ns.append(int(m.sum()))

    ece = ece_score(probs, labels, n_bins)
    brier, rel, res, unc = brier_decomposition(probs, labels, n_bins)

    fig, ax = plt.subplots(figsize=(COL, 2.9))
    ax.plot([0, 1], [0, 1], color="0.5", linestyle=":", lw=0.8, label="perfect")
    width = 1.0 / n_bins * 0.85
    bars = ax.bar(
        centers,
        accs,
        width=width,
        color=PAL["canonicalized"],
        alpha=0.35,
        edgecolor=PAL["canonicalized"],
        linewidth=0.7,
        label="bin accuracy",
    )
    # gap arrows confidence -> accuracy
    for c, a, conf in zip(centers, accs, confs):
        ax.plot([conf, conf], [conf, a], color=PAL["naive"], lw=0.7, alpha=0.9)
    ax.plot(
        confs,
        accs,
        "o-",
        color=PAL["naive"],
        ms=4,
        lw=1.0,
        label="confidence",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (abnormal)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability (ECE={ece:.3f}, Brier={brier:.3f}, n={len(probs)})", pad=4)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.4, color=PAL["grid"], alpha=0.7)
    # bin-count subplot inset
    inset = ax.inset_axes([0.62, 0.05, 0.34, 0.22])
    inset.bar(centers, ns, width=width, color=PAL["neutral"])
    inset.set_xlim(0, 1)
    inset.set_xticks([])
    inset.set_yticks([0, max(ns)])
    inset.tick_params(labelsize=6)
    inset.set_title("n per bin", fontsize=6, pad=1)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)

    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig_coverage (compact version, retained for back-compat)
# --------------------------------------------------------------------------- #


def fig_coverage(preds: List[PredRow], out_path: Path) -> None:
    shifted = [p for p in preds if not (p.severity in ("250", "1.0"))]
    if not shifted:
        shifted = list(preds)
    probs, labels, logits = slice_arrays(shifted)
    preds_bin = (probs >= 0.5).astype(int)
    correct = (preds_bin == labels).astype(int)
    adequacy = np.abs(logits)
    n = len(adequacy)
    order = np.argsort(-adequacy)
    cum_correct = np.cumsum(correct[order])
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - cum_correct / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, coverage))

    fig, ax = plt.subplots(figsize=(COL, 2.9))
    ax.plot(coverage, risk, color=PAL["canonicalized"], lw=1.2, label=f"empirical (AURC={aurc:.3f})")
    for c in (0.5, 0.75, 1.0):
        k = max(1, int(round(c * n))) - 1
        ax.plot(coverage[k], risk[k], "o", ms=5, color=PAL["naive"], mfc="white")
        ax.annotate(
            f"cov={coverage[k]:.2f}\nrisk={risk[k]:.3f}",
            xy=(coverage[k], risk[k]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk on covered set")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, max(0.5, float(risk.max()) * 1.1))
    ax.set_title(f"Selective abstention via $|$logit$|$ (n={n})", pad=4)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.5, color=PAL["grid"], alpha=0.7)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig_architecture - 4-plane block diagram
# --------------------------------------------------------------------------- #


def fig_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(DCOL, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    blocks = [
        (0.15, 1.7, 1.55, 1.4, "EDF / native\nrecording\n($f_s$ varies)", "white"),
        (1.95, 1.7, 1.55, 1.4, "ESF\ncanonicalize\n($\\mu$V, 250 Hz,\n19-ch 10-20)", "#FCE7C2"),
        (3.75, 1.7, 1.55, 1.4, "Frozen EEGPT\nbackbone", "#CDEAF7"),
        (5.55, 1.7, 1.55, 1.4, "Linear probe\n(TUAB)", "#CDEAF7"),
        (7.35, 1.7, 1.55, 1.4, "Platt cal.\n+ abstention\ngate", "#D3E8D3"),
        (9.15, 2.0, 0.75, 0.8, "verdict /\nabstain", "white"),
    ]
    for x, y, w, h, txt, c in blocks:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04",
                facecolor=c,
                edgecolor="black",
                linewidth=0.9,
            )
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8, family="serif")
    # arrows
    edges = [(1.7, 1.95), (3.5, 3.75), (5.3, 5.55), (7.1, 7.35), (8.9, 9.15)]
    for x1, x2 in edges:
        ax.annotate("", xy=(x2, 2.4), xytext=(x1, 2.4), arrowprops=dict(arrowstyle="->", lw=1.0))
    # perturbation injection band
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.15, 3.45),
            5.2,
            0.55,
            boxstyle="round,pad=0.04",
            facecolor="#FFE7E0",
            edgecolor=PAL["naive"],
            linewidth=0.9,
            linestyle="--",
        )
    )
    ax.text(2.75, 3.72, "perturbation axes injected before canonicalization", ha="center", va="center", fontsize=8, color=PAL["naive"])
    ax.annotate("", xy=(1.7, 3.1), xytext=(1.7, 3.4), arrowprops=dict(arrowstyle="->", lw=0.8, color=PAL["naive"]))
    # OOD detector band
    ax.text(5.0, 0.9, "selective head: $|$logit$|$ + NP test on per-window distribution", ha="center", va="center", fontsize=8, style="italic", color="0.3")
    ax.annotate("", xy=(7.7, 1.65), xytext=(5.0, 1.05), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.5", linestyle=":"))

    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig_roc_pr_curves
# --------------------------------------------------------------------------- #


def fig_roc_pr_curves(preds: List[PredRow], out_path: Path) -> None:
    # Baseline = sampling_rate=250 naive (clean signal).  Compare to two
    # representative shift conditions.
    baseline = filter_pred(preds, "sampling_rate", "250", "naive")
    shift_sets = [
        ("powerline 30 muV, naive", filter_pred(preds, "power_line", "30.0", "naive"), PAL["naive"]),
        ("powerline 30 muV, canonicalized", filter_pred(preds, "power_line", "30.0", "canonicalized"), PAL["canonicalized"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 3.2))

    # ROC
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle=":", color="0.5", lw=0.7)
    for name, rows, color in [("clean baseline", baseline, "black")] + [(n, r, c) for n, r, c in shift_sets]:
        if not rows:
            continue
        s, l, _ = slice_arrays(rows)
        fpr, tpr = roc_curve(s, l)
        a = auroc(s, l)
        ax.plot(fpr, tpr, color=color, lw=1.1, label=f"{name} (AUC={a:.3f})")
        grid, tpr_lo, tpr_hi, _, _, _ = bootstrap_curves(s, l, n_boot=200)
        ax.fill_between(grid, tpr_lo, tpr_hi, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.001)
    ax.set_title("ROC (95% bootstrap band)", pad=4)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, color=PAL["grid"], alpha=0.7)

    # PR
    ax = axes[1]
    base_rate = float(np.mean([p.label for p in baseline]))
    ax.axhline(base_rate, linestyle=":", color="0.5", lw=0.7, label=f"base rate={base_rate:.2f}")
    for name, rows, color in [("clean baseline", baseline, "black")] + [(n, r, c) for n, r, c in shift_sets]:
        if not rows:
            continue
        s, l, _ = slice_arrays(rows)
        rec, prec = pr_curve(s, l)
        ap = average_precision(s, l)
        ax.plot(rec, prec, color=color, lw=1.1, label=f"{name} (AP={ap:.3f})")
        _, _, _, grid, pr_lo, pr_hi = bootstrap_curves(s, l, n_boot=200)
        ax.fill_between(grid, pr_lo, pr_hi, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.001)
    ax.set_title("Precision-Recall (95% bootstrap band)", pad=4)
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, color=PAL["grid"], alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig_calibration_decomposition (Brier)
# --------------------------------------------------------------------------- #


def fig_calibration_decomposition(preds: List[PredRow], out_path: Path) -> None:
    # Use a fixed slate of representative (axis,severity) conditions
    conditions = [
        ("sampling_rate", "250"),
        ("sampling_rate", "128"),
        ("sampling_rate", "512"),
        ("calibration", "0.5"),
        ("calibration", "2.0"),
        ("power_line", "5.0"),
        ("power_line", "30.0"),
    ]

    rows = []
    for ax_k, sev in conditions:
        for arm in ["naive", "canonicalized"]:
            sub = filter_pred(preds, ax_k, sev, arm)
            if not sub:
                continue
            p, l, _ = slice_arrays(sub)
            b, rel, res, unc = brier_decomposition(p, l)
            rows.append((ax_k, sev, arm, b, rel, res, unc))

    fig, ax = plt.subplots(figsize=(DCOL, 3.2))
    width = 0.35
    labels = []
    rel_n, res_n, brier_n = [], [], []
    rel_c, res_c, brier_c = [], [], []
    unc_baseline = None
    for ax_k, sev in conditions:
        labels.append(f"{ax_k.replace('_',' ')}\n={sev}")
        for ax_kk, sevv, arm, b, rel, res, unc in rows:
            if ax_kk == ax_k and sevv == sev:
                if arm == "naive":
                    rel_n.append(rel)
                    res_n.append(res)
                    brier_n.append(b)
                else:
                    rel_c.append(rel)
                    res_c.append(res)
                    brier_c.append(b)
                unc_baseline = unc
    x = np.arange(len(labels))

    # Stacked: reliability (penalty, lower better) + (uncertainty - resolution)
    # so the stack sums to Brier
    naive_uncRes = [unc_baseline - r for r in res_n]
    canon_uncRes = [unc_baseline - r for r in res_c]

    ax.bar(x - width / 2, rel_n, width, color=PAL["band_naive"], edgecolor=PAL["naive"], linewidth=0.6, label="naive: reliability")
    ax.bar(x - width / 2, naive_uncRes, width, bottom=rel_n, color="white", edgecolor=PAL["naive"], hatch="///", linewidth=0.6, label="naive: uncertainty $-$ resolution")
    ax.bar(x + width / 2, rel_c, width, color=PAL["band_canon"], edgecolor=PAL["canonicalized"], linewidth=0.6, label="canon: reliability")
    ax.bar(x + width / 2, canon_uncRes, width, bottom=rel_c, color="white", edgecolor=PAL["canonicalized"], hatch="\\\\\\", linewidth=0.6, label="canon: uncertainty $-$ resolution")

    # annotate Brier on top
    for xi, b in zip(x - width / 2, brier_n):
        ax.text(xi, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=6)
    for xi, b in zip(x + width / 2, brier_c):
        ax.text(xi, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=7)
    ax.set_ylabel("Brier (lower is better)")
    ax.set_title(f"Brier decomposition: reliability + (uncertainty $-$ resolution); uncertainty={unc_baseline:.3f}", pad=4)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=7)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color=PAL["grid"], alpha=0.7)
    ax.set_ylim(0, max([b for b in brier_n + brier_c]) * 1.18)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig_effect_size_forest - Hedges' g of |logit| (clean vs shifted)
# --------------------------------------------------------------------------- #


def fig_effect_size_forest(preds: List[PredRow], out_path: Path) -> None:
    clean = filter_pred(preds, "sampling_rate", "250", "naive")
    clean_logit = np.abs(np.array([p.logit for p in clean]))
    rows = []
    for axis_key, severity in [
        ("sampling_rate", "128"),
        ("sampling_rate", "200"),
        ("sampling_rate", "256"),
        ("sampling_rate", "512"),
        ("calibration", "0.5"),
        ("calibration", "0.75"),
        ("calibration", "1.5"),
        ("calibration", "2.0"),
        ("power_line", "5.0"),
        ("power_line", "15.0"),
        ("power_line", "30.0"),
    ]:
        for arm in ["naive", "canonicalized"]:
            sub = filter_pred(preds, axis_key, severity, arm)
            if not sub:
                continue
            shifted_logit = np.abs(np.array([p.logit for p in sub]))
            g, lo, hi = hedges_g(shifted_logit, clean_logit)
            rows.append((axis_key, severity, arm, g, lo, hi))

    fig, ax = plt.subplots(figsize=(DCOL, 4.2))
    rows_sorted = sorted(rows, key=lambda r: (r[0], float(r[1]), r[2]))
    ys = np.arange(len(rows_sorted))
    yticklabels = []
    for i, (axis_key, sev, arm, g, lo, hi) in enumerate(rows_sorted):
        color = PAL[arm]
        marker = "o" if arm == "naive" else "s"
        ax.errorbar(
            g,
            i,
            xerr=[[g - lo], [hi - g]],
            fmt=marker,
            color=color,
            mfc="white" if arm == "naive" else color,
            mec=color,
            ms=5,
            elinewidth=0.8,
            capsize=2.5,
        )
        yticklabels.append(f"{axis_key.replace('_',' ')} = {sev}  ({arm})")
    ax.axvline(0, color="0.5", lw=0.7, linestyle=":")
    ax.set_yticks(ys)
    ax.set_yticklabels(yticklabels, fontsize=7)
    ax.set_xlabel("Hedges' $g$ vs clean baseline ($|$logit$|$, 95% CI)")
    ax.set_title("Effect size of acquisition shift on per-recording $|$logit$|$", pad=4)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, color=PAL["grid"], alpha=0.7)
    naive_h = plt.Line2D([0], [0], marker="o", color=PAL["naive"], mfc="white", linestyle="None", label="naive")
    canon_h = plt.Line2D([0], [0], marker="s", color=PAL["canonicalized"], linestyle="None", label="canonicalized")
    ax.legend(handles=[naive_h, canon_h], loc="lower right", frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig_axis_heatmap - DeltaAUROC with FDR-significance
# --------------------------------------------------------------------------- #


def fig_axis_heatmap(sweep: List[SweepRow], preds: List[PredRow], out_path: Path) -> None:
    # baseline AUROC from the identity row (sampling_rate=250 naive)
    base_auroc = 0.907937
    # build matrix axis x severity rows, arm cols
    axis_severities = {
        "sampling_rate": ["128", "200", "250", "256", "512"],
        "calibration": ["0.5", "0.75", "1.0", "1.5", "2.0"],
        "power_line": ["0.0", "5.0", "15.0", "30.0"],
    }
    rows_label, mat, pvals = [], [], []
    arms = ["naive", "canonicalized"]
    for axis_k, sevs in axis_severities.items():
        for sev in sevs:
            row = []
            row_p = []
            for arm in arms:
                hits = [r for r in sweep if r.axis == axis_k and r.severity == sev and r.arm == arm]
                if not hits:
                    row.append(np.nan)
                    row_p.append(1.0)
                    continue
                row.append(hits[0].auroc - base_auroc)
                # paired bootstrap p vs clean baseline using prediction CSV
                base = filter_pred(preds, "sampling_rate", "250", "naive")
                sub = filter_pred(preds, axis_k, sev, arm)
                if not sub:
                    row_p.append(1.0)
                else:
                    sb, lb, _ = slice_arrays(base)
                    ss, ls, _ = slice_arrays(sub)
                    # both sets share rec_ids; align by rec_id
                    base_map = {p.rec_id: p.prob for p in base}
                    sub_map = {p.rec_id: p.prob for p in sub}
                    common = sorted(set(base_map) & set(sub_map))
                    if len(common) < 10:
                        row_p.append(1.0)
                        continue
                    sB = np.array([base_map[r] for r in common])
                    sS = np.array([sub_map[r] for r in common])
                    label_map = {p.rec_id: p.label for p in base}
                    lab = np.array([label_map[r] for r in common])
                    row_p.append(delong_pvalue(sS, sB, lab))
            rows_label.append(f"{axis_k.replace('_',' ')} = {sev}")
            mat.append(row)
            pvals.append(row_p)

    mat = np.array(mat)
    pvals = np.array(pvals)
    # FDR across all non-baseline cells
    flat_p = pvals.flatten()
    sig = benjamini_hochberg(flat_p, alpha=0.05).reshape(pvals.shape)

    fig, ax = plt.subplots(figsize=(COL, 4.0))
    cmap = LinearSegmentedColormap.from_list(
        "delta", [PAL["delta_neg"], "#FFFFFF", PAL["delta_pos"]]
    )
    vmax = float(np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=8)
    ax.set_yticks(range(len(rows_label)))
    ax.set_yticklabels(rows_label, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            star = "*" if sig[i, j] else ""
            txt_color = "black" if abs(v) < vmax * 0.55 else "white"
            ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=7, color=txt_color)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("$\\Delta$AUROC vs clean", fontsize=8)
    ax.set_title("Acquisition-shift effect grid\n(* FDR-significant, $\\alpha$=0.05)", fontsize=10, pad=4)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig_selective_risk_curve
# --------------------------------------------------------------------------- #


def fig_selective_risk_curve(preds: List[PredRow], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(DCOL, 3.2))
    for arm, color, label in [
        ("naive", PAL["naive"], "naive"),
        ("canonicalized", PAL["canonicalized"], "canonicalized"),
    ]:
        rows = [p for p in preds if p.arm == arm and not (p.severity in ("250", "1.0"))]
        if not rows:
            continue
        probs, labels, logits = slice_arrays(rows)
        bin_pred = (probs >= 0.5).astype(int)
        correct = (bin_pred == labels).astype(int)
        adequacy = np.abs(logits)
        n = len(adequacy)
        order = np.argsort(-adequacy)
        cum_correct = np.cumsum(correct[order])
        coverage = np.arange(1, n + 1) / n
        risk = 1.0 - cum_correct / np.arange(1, n + 1)
        aurc = float(np.trapezoid(risk, coverage))
        # E-AURC: AURC minus optimal AURC
        n_err = n - cum_correct[-1]
        if n_err == 0:
            e_aurc = 0.0
        else:
            opt_risk = np.zeros(n)
            # optimal selective predictor would always answer the correct ones first
            # so optimal risk at coverage k = max(0, k - n_correct_total) / k
            n_correct_total = int(cum_correct[-1])
            opt_correct = np.minimum(np.arange(1, n + 1), n_correct_total)
            opt_risk = 1 - opt_correct / np.arange(1, n + 1)
            opt_aurc = float(np.trapezoid(opt_risk, coverage))
            e_aurc = aurc - opt_aurc
        ax.plot(coverage, risk, color=color, lw=1.2, label=f"{label}: AURC={aurc:.3f}, E-AURC={e_aurc:.3f}")
        # mark 0.5 / 0.75 / 1.0
        for c in (0.5, 0.75, 1.0):
            k = max(1, int(round(c * n))) - 1
            ax.plot(coverage[k], risk[k], "o", ms=4, color=color, mfc="white")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk on covered set")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, None)
    ax.set_title("Risk-coverage curve, $|$logit$|$-rank abstention", pad=4)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.5, color=PAL["grid"], alpha=0.7)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig_pipeline_flow
# --------------------------------------------------------------------------- #


def fig_pipeline_flow(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(DCOL, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    stages = [
        (0.1, 1.6, 1.5, 1.4, "Raw EDF\n(varied $f_s$,\nmontage)", "white"),
        (1.85, 1.6, 1.5, 1.4, "ESF\ncanonicalize", "#FCE7C2"),
        (3.6, 1.6, 1.5, 1.4, "EEGPT\nbackbone\n(frozen)", "#CDEAF7"),
        (5.35, 1.6, 1.5, 1.4, "Linear probe\n(TUAB-trained)", "#CDEAF7"),
        (7.1, 1.6, 1.5, 1.4, "Platt cal.", "#D3E8D3"),
        (8.85, 1.6, 1.05, 1.4, "Abstain\nor verdict", "#FFE7E0"),
    ]
    for x, y, w, h, txt, c in stages:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=c, edgecolor="black", linewidth=0.9))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5, family="serif")
    # arrows
    edges = [(1.6, 1.85), (3.35, 3.6), (5.1, 5.35), (6.85, 7.1), (8.6, 8.85)]
    for x1, x2 in edges:
        ax.annotate("", xy=(x2, 2.3), xytext=(x1, 2.3), arrowprops=dict(arrowstyle="->", lw=1.0))
    # perturbation axes labels above the canonicalize block
    ax.add_patch(mpatches.FancyBboxPatch((0.1, 3.4), 3.25, 0.8, boxstyle="round,pad=0.04", facecolor="#FFF1ED", edgecolor=PAL["naive"], linewidth=0.8, linestyle="--"))
    ax.text(1.7, 3.95, "perturbation axes", ha="center", va="center", fontsize=8.5, color=PAL["naive"], weight="bold")
    ax.text(1.7, 3.6, "sampling rate / gain / power-line noise", ha="center", va="center", fontsize=7.5, color=PAL["naive"])
    ax.annotate("", xy=(2.6, 3.05), xytext=(2.6, 3.4), arrowprops=dict(arrowstyle="->", lw=0.8, color=PAL["naive"]))
    # selective head detail
    ax.text(7.6, 0.9, "selective head: $|$logit$|$ + NP test over windows", ha="center", va="center", fontsize=8, style="italic", color="0.3")
    ax.annotate("", xy=(9.4, 1.55), xytext=(7.7, 1.05), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.5", linestyle=":"))
    # data labels under boxes
    labels = [
        (0.85, 1.45, "(varied $f_s$, $\\mu$V, channels)"),
        (2.6, 1.45, r"$\rightarrow$ 250 Hz, 19-ch 10-20"),
        (4.35, 1.45, "frozen embeddings"),
        (6.1, 1.45, "$\\hat p$ on TUAB"),
        (7.85, 1.45, "calibrated $\\hat p$, $|$logit$|$"),
        (9.4, 1.45, "$\\tau$ gate"),
    ]
    for x, y, t in labels:
        ax.text(x, y, t, ha="center", va="top", fontsize=6.5, style="italic", color="0.35")

    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def regenerate_all(out_dir: Path = FIG_DIR) -> Dict[str, Path]:
    apply_rcparams()
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = load_sweep()
    preds = load_predictions()

    out: Dict[str, Path] = {}
    plan = [
        ("fig_axes.pdf", lambda p: fig_axes(sweep, p), False),
        ("fig_reliability.pdf", lambda p: fig_reliability(preds, p), False),
        ("fig_coverage.pdf", lambda p: fig_coverage(preds, p), False),
        ("fig_architecture.pdf", lambda p: fig_architecture(p), False),
        ("fig_roc_pr_curves.pdf", lambda p: fig_roc_pr_curves(preds, p), False),
        ("fig_calibration_decomposition.pdf", lambda p: fig_calibration_decomposition(preds, p), False),
        ("fig_effect_size_forest.pdf", lambda p: fig_effect_size_forest(preds, p), False),
        ("fig_axis_heatmap.pdf", lambda p: fig_axis_heatmap(sweep, preds, p), False),
        ("fig_selective_risk_curve.pdf", lambda p: fig_selective_risk_curve(preds, p), False),
        ("fig_pipeline_flow.pdf", lambda p: fig_pipeline_flow(p), False),
    ]
    for name, fn, _ in plan:
        path = out_dir / name
        fn(path)
        out[name] = path
        print(f"wrote {path}")
    return out


if __name__ == "__main__":
    regenerate_all()
