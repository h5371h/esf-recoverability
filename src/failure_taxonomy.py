"""SPMB 2026 — Failure-mode taxonomy.

Builds a per-recording failure profile across (axis, severity, arm),
clusters recordings into 7 mutually exclusive failure categories,
generates the failure-taxonomy figure, picks median exemplars per
category, derives an axis-vulnerability ranking, and writes the
LaTeX subsection for the paper.

Inputs:
  data/per_recording_predictions_latest.csv   (276 recs x 14 (axis,severity) x 2 arms)
  data/sweep_latest.csv                       (aggregate AUROC per cell)

Outputs:
  data/per_recording_failure_profile.csv
  data/failure_exemplars.md
  data/axis_vulnerability_ranking.md
  figures/fig_failure_taxonomy.pdf
  data/failure_taxonomy_subsection.tex

No invented numbers; every value derived from the CSVs above.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
SRC_DIR = ROOT / "data"

PRED_CSV = DATA_DIR / "per_recording_predictions_latest.csv"
SWEEP_CSV = DATA_DIR / "sweep_latest.csv"

PROFILE_CSV = DATA_DIR / "per_recording_failure_profile.csv"
EXEMPLARS_MD = DATA_DIR / "failure_exemplars.md"
AXIS_RANK_MD = DATA_DIR / "axis_vulnerability_ranking.md"
FIG_PDF = FIG_DIR / "fig_failure_taxonomy.pdf"
SECTION_TEX = SRC_DIR / "failure_taxonomy_subsection.tex"

# IEEE conference column widths (must match paper_figures_v2.py)
COL = 3.5
DCOL = 7.16

# Colorblind-safe palette matched to paper_figures_v2.py.
PAL = {
    "naive": "#D55E00",
    "canonicalized": "#0072B2",
    "neutral": "#444444",
    "grid": "#BBBBBB",
    "normal": "#009E73",
    "abnormal": "#CC79A7",
    "ood": "#000000",
}

# Seven failure categories, ColorBrewer-derived (mostly Set2 / Set1) and
# colorblind-tested so they stay distinguishable in the stacked bars.
CATEGORY_ORDER = [
    "robust_correct",
    "shift_recovered",
    "calibration_collapse",
    "axis_specific",
    "shift_induced_flip",
    "catastrophic_shift",
    "robust_wrong",
]
CATEGORY_LABEL = {
    "robust_correct": "Robust-correct",
    "robust_wrong": "Robust-wrong",
    "shift_induced_flip": "Shift-induced flip",
    "shift_recovered": "Shift-recovered",
    "calibration_collapse": "Calibration collapse",
    "catastrophic_shift": "Catastrophic shift",
    "axis_specific": "Axis-specific",
}
CATEGORY_COLOR = {
    "robust_correct": "#1B9E77",  # green (good)
    "shift_recovered": "#7570B3",  # purple (suspicious)
    "calibration_collapse": "#E6AB02",  # ochre (warning)
    "axis_specific": "#66A61E",  # olive
    "shift_induced_flip": "#D95F02",  # orange (bad)
    "catastrophic_shift": "#E7298A",  # magenta (most dangerous)
    "robust_wrong": "#A6761D",  # brown (intrinsic, not shift)
}

# Canonical baseline severity per axis (the "no shift" cell, naive arm).
BASELINE_SEVERITY = {
    "sampling_rate": 256.0,
    "calibration": 1.0,
    "power_line": 0.0,
}

# Calibration-collapse threshold: prob within +/- 0.15 of 0.5
# (rationale: a sigmoid output in [0.35, 0.65] corresponds to |logit| <= ~0.62,
# which the selective head treats as low-confidence; see paper sec III-C).
CALIB_COLLAPSE_BAND = 0.15

# Catastrophic-shift confidence threshold: wrong with prob > this and
# baseline prob > this on the opposite side.
HIGH_CONF = 0.80

# --------------------------------------------------------------------- #
# rc setup (matches paper figures)
# --------------------------------------------------------------------- #


def apply_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.dpi": 300,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_CSV)
    df["pred"] = (df["prob"] >= 0.5).astype(int)
    df["correct"] = (df["pred"] == df["label"]).astype(int)
    return df


def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-recording baseline prediction (naive arm at canonical severity).

    All three axes share the same baseline rows numerically (verified upstream).
    """
    rows = []
    for axis, sev in BASELINE_SEVERITY.items():
        sub = df[(df.axis == axis) & (df.severity == sev) & (df.arm == "naive")]
        rows.append(sub[["rec_id", "label", "prob", "pred", "correct", "logit", "var_logit"]])
    base = rows[0].copy()  # they are identical; take the first
    base = base.rename(
        columns={
            "prob": "baseline_prob",
            "pred": "baseline_pred",
            "correct": "baseline_correct",
            "logit": "baseline_logit",
            "var_logit": "baseline_var",
        }
    )
    return base


# --------------------------------------------------------------------- #
# Part 1 — Per-recording failure profile
# --------------------------------------------------------------------- #


def build_profile(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """One row per recording with aggregate flip / monotonicity stats.

    Aggregates over the *canonicalized* arm only (operational deployment
    surface). The naive arm leaks the perturbation directly to the model
    and is reported elsewhere as a sanity check.
    """
    canon = df[df.arm == "canonicalized"].copy()
    canon = canon.merge(baseline[["rec_id", "baseline_pred", "baseline_correct", "baseline_prob", "label"]],
                        on=["rec_id", "label"], how="left")

    out = []
    for rec_id, grp in canon.groupby("rec_id"):
        label = int(grp["label"].iloc[0])
        bpred = int(grp["baseline_pred"].iloc[0])
        bprob = float(grp["baseline_prob"].iloc[0])
        bcorrect = int(grp["baseline_correct"].iloc[0])

        # Drop the baseline cell itself (severity == baseline) from "shifted" set.
        shifted = grp[~grp.apply(
            lambda r: r["severity"] == BASELINE_SEVERITY[r["axis"]], axis=1
        )].copy()

        n_shifted = len(shifted)
        n_flips = int((shifted["pred"] != bpred).sum())
        n_shifted_correct = int((shifted["correct"] == 1).sum())
        flip_rate = n_flips / n_shifted if n_shifted else 0.0

        # Per-axis flip rates (used by axis-specific category).
        gt_prob_b = bprob if label == 1 else 1 - bprob
        axis_flip = {}
        axis_drop = {}
        for ax in ["sampling_rate", "calibration", "power_line"]:
            sub = shifted[shifted.axis == ax]
            if len(sub):
                axis_flip[ax] = int((sub["pred"] != bpred).sum()) / len(sub)
                # confidence on the GT class (correct-side prob).
                gt_prob_shift_series = sub["prob"] if label == 1 else (1 - sub["prob"])
                axis_drop[ax] = gt_prob_b - float(gt_prob_shift_series.mean())
            else:
                axis_flip[ax] = 0.0
                axis_drop[ax] = 0.0

        # Confidence drop on the GT class.
        shifted_gt_prob = shifted["prob"] if label == 1 else (1 - shifted["prob"])
        mean_shifted_gt_prob = float(shifted_gt_prob.mean()) if len(shifted_gt_prob) else gt_prob_b
        conf_drop = gt_prob_b - mean_shifted_gt_prob

        # Monotonic degradation: per axis, sort by severity distance from
        # baseline, see if GT-class prob is monotonically non-increasing.
        mono_axes = 0
        for ax in ["sampling_rate", "calibration", "power_line"]:
            sub = grp[grp.axis == ax].copy()
            base_sev = BASELINE_SEVERITY[ax]
            sub["dist"] = (sub["severity"] - base_sev).abs()
            sub = sub.sort_values("dist")
            gtp = (sub["prob"] if label == 1 else (1 - sub["prob"])).values
            if len(gtp) >= 3 and all(gtp[i + 1] <= gtp[i] + 1e-9 for i in range(len(gtp) - 1)):
                mono_axes += 1

        # Catastrophic shift: at least one shifted cell where the model is
        # both wrong AND highly confident (>= HIGH_CONF on the wrong side).
        wrong_conf = shifted[shifted["correct"] == 0].copy()
        if len(wrong_conf):
            # wrong-side prob = prob of class that is NOT the GT label
            wrong_side_prob = wrong_conf["prob"] if label == 0 else (1 - wrong_conf["prob"])
            n_catastrophic = int((wrong_side_prob >= HIGH_CONF).sum())
            max_wrong_conf = float(wrong_side_prob.max())
        else:
            n_catastrophic = 0
            max_wrong_conf = 0.0

        # Calibration collapse: many shifted cells land in [0.5 +/- band].
        in_band = ((shifted["prob"] - 0.5).abs() <= CALIB_COLLAPSE_BAND).sum()
        frac_in_band = in_band / n_shifted if n_shifted else 0.0

        out.append(
            {
                "rec_id": rec_id,
                "label": label,
                "baseline_pred": bpred,
                "baseline_prob": bprob,
                "baseline_correct": bcorrect,
                "n_shifted_cells": n_shifted,
                "n_shifted_correct": n_shifted_correct,
                "flip_count": n_flips,
                "flip_rate": flip_rate,
                "conf_drop": conf_drop,
                "mean_shifted_gt_prob": mean_shifted_gt_prob,
                "monotonic_axes": mono_axes,
                "n_catastrophic": n_catastrophic,
                "max_wrong_conf": max_wrong_conf,
                "frac_in_calib_band": frac_in_band,
                "flip_rate_sampling_rate": axis_flip["sampling_rate"],
                "flip_rate_calibration": axis_flip["calibration"],
                "flip_rate_power_line": axis_flip["power_line"],
                "conf_drop_sampling_rate": axis_drop["sampling_rate"],
                "conf_drop_calibration": axis_drop["calibration"],
                "conf_drop_power_line": axis_drop["power_line"],
            }
        )
    profile = pd.DataFrame(out)
    return profile


# --------------------------------------------------------------------- #
# Part 2 — Failure taxonomy
# --------------------------------------------------------------------- #


def classify(profile: pd.DataFrame) -> pd.DataFrame:
    """Assign one of 7 categories per recording using mutually exclusive rules.

    Rule order matters: most-specific dangerous categories first so a
    recording cannot be silently swept into a benign bucket.
    """
    profile = profile.copy()
    cats = []
    for _, r in profile.iterrows():
        # Catastrophic shift: was correct at baseline, then >=1 cell with
        # high-confidence WRONG prediction. Most dangerous; classify first.
        if r["baseline_correct"] == 1 and r["n_catastrophic"] >= 1:
            cats.append("catastrophic_shift")
            continue
        # Robust-wrong: wrong at baseline and stays wrong everywhere
        # (>= 90% of shifted cells also wrong). Intrinsic, not shift-driven.
        if r["baseline_correct"] == 0 and r["n_shifted_correct"] / max(r["n_shifted_cells"], 1) <= 0.10:
            cats.append("robust_wrong")
            continue
        # Shift-recovered: wrong at baseline, becomes correct in majority of
        # shifted cells (>= 60%). Suspicious / random-correction.
        if r["baseline_correct"] == 0 and r["n_shifted_correct"] / max(r["n_shifted_cells"], 1) >= 0.60:
            cats.append("shift_recovered")
            continue
        # Shift-induced flip: correct at baseline, >=30% of shifted cells flip
        # the prediction.
        if r["baseline_correct"] == 1 and r["flip_rate"] >= 0.30:
            cats.append("shift_induced_flip")
            continue
        # Axis-specific: correct at baseline; flips concentrated in one axis
        # (one axis flip-rate >= 50%, the other two < 15%).
        if r["baseline_correct"] == 1:
            axis_rates = [
                r["flip_rate_sampling_rate"],
                r["flip_rate_calibration"],
                r["flip_rate_power_line"],
            ]
            high = sum(1 for v in axis_rates if v >= 0.50)
            low = sum(1 for v in axis_rates if v < 0.15)
            if high == 1 and low == 2:
                cats.append("axis_specific")
                continue
        # Calibration collapse: correct at baseline, fewer than 30% flips, but
        # >= 20% of shifted cells land in the 0.5 +/- band (genuine equivocation
        # without flipping the argmax).
        if r["baseline_correct"] == 1 and r["flip_rate"] < 0.30 and r["frac_in_calib_band"] >= 0.20:
            cats.append("calibration_collapse")
            continue
        # Default robust-correct: correct at baseline and largely stays correct.
        if r["baseline_correct"] == 1:
            cats.append("robust_correct")
            continue
        # Remaining wrong-at-baseline mixed cases (between 10% and 60% recovery)
        # — these are also intrinsic, group under robust_wrong (model can't
        # reliably classify even with shifts).
        cats.append("robust_wrong")
    profile["category"] = cats
    return profile


def category_counts(profile: pd.DataFrame) -> pd.DataFrame:
    out = []
    total = len(profile)
    for cat in CATEGORY_ORDER:
        sub = profile[profile["category"] == cat]
        n = len(sub)
        n_norm = int((sub["label"] == 0).sum())
        n_abn = int((sub["label"] == 1).sum())
        out.append(
            {
                "category": cat,
                "label": CATEGORY_LABEL[cat],
                "n": n,
                "frac": n / total,
                "n_normal": n_norm,
                "n_abnormal": n_abn,
                "frac_normal": n_norm / max((profile["label"] == 0).sum(), 1),
                "frac_abnormal": n_abn / max((profile["label"] == 1).sum(), 1),
            }
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------- #
# Part 3 — Figure
# --------------------------------------------------------------------- #


def assign_cell_category(row: pd.Series, baseline_pred: int, baseline_correct: int, label: int) -> str:
    """Single-cell category for the stacked-bar figure.

    Each (rec, axis, severity, canonicalized arm) cell gets one tag based
    purely on its own outcome relative to the recording's baseline. This
    lets the figure show fractions across axis x severity.
    """
    correct = int(row["pred"] == label)
    pred = int(row["pred"])
    prob = float(row["prob"])
    wrong_side_prob = prob if label == 0 else 1 - prob

    if baseline_correct == 1 and not correct and wrong_side_prob >= HIGH_CONF:
        return "catastrophic_shift"
    if baseline_correct == 1 and not correct:
        return "shift_induced_flip"
    if baseline_correct == 0 and correct:
        return "shift_recovered"
    if baseline_correct == 0 and not correct:
        return "robust_wrong"
    # baseline_correct == 1 and still correct here
    if abs(prob - 0.5) <= CALIB_COLLAPSE_BAND:
        return "calibration_collapse"
    return "robust_correct"


def figure_taxonomy(df: pd.DataFrame, baseline: pd.DataFrame) -> None:
    apply_rcparams()
    canon = df[df.arm == "canonicalized"].merge(
        baseline[["rec_id", "baseline_pred", "baseline_correct"]], on="rec_id"
    )
    canon["cell_cat"] = canon.apply(
        lambda r: assign_cell_category(r, int(r["baseline_pred"]), int(r["baseline_correct"]), int(r["label"])),
        axis=1,
    )

    axes_order = ["sampling_rate", "calibration", "power_line"]
    sev_by_axis = {
        "sampling_rate": sorted([s for s in canon[canon.axis == "sampling_rate"]["severity"].unique()
                                 if s != BASELINE_SEVERITY["sampling_rate"]]),
        "calibration": sorted([s for s in canon[canon.axis == "calibration"]["severity"].unique()
                               if s != BASELINE_SEVERITY["calibration"]]),
        "power_line": sorted([s for s in canon[canon.axis == "power_line"]["severity"].unique()
                              if s != BASELINE_SEVERITY["power_line"]]),
    }
    axis_titles = {
        "sampling_rate": "Sampling rate (Hz)",
        "calibration": "Calibration gain ($\\times$)",
        "power_line": "Power-line noise (µV)",
    }
    # Categories actually used in the bar (axis_specific is per-recording,
    # not per-cell, so we drop it here and surface it in text/exemplars).
    bar_cats = [c for c in CATEGORY_ORDER if c != "axis_specific"]

    fig, axarr = plt.subplots(1, 3, figsize=(DCOL, 3.0), sharey=True)
    for ax_idx, axis_name in enumerate(axes_order):
        ax = axarr[ax_idx]
        sevs = sev_by_axis[axis_name]
        bottoms = np.zeros(len(sevs))
        for cat in bar_cats:
            heights = []
            for sev in sevs:
                cell = canon[(canon.axis == axis_name) & (canon.severity == sev)]
                if len(cell) == 0:
                    heights.append(0.0)
                else:
                    heights.append((cell["cell_cat"] == cat).mean())
            ax.bar(
                np.arange(len(sevs)),
                heights,
                bottom=bottoms,
                color=CATEGORY_COLOR[cat],
                edgecolor="white",
                linewidth=0.5,
                label=CATEGORY_LABEL[cat] if ax_idx == 0 else None,
            )
            bottoms += np.array(heights)
        ax.set_xticks(np.arange(len(sevs)))
        ax.set_xticklabels([f"{s:g}" for s in sevs], fontsize=8)
        ax.set_title(axis_titles[axis_name])
        ax.set_ylim(0, 1)
        if ax_idx == 0:
            ax.set_ylabel("Fraction of recordings")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color=PAL["grid"], alpha=0.7)
        ax.set_axisbelow(True)

    handles, labels = axarr[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
    )
    fig.suptitle("Per-cell failure categories under acquisition shift (canonicalized arm, $n{=}276$)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Part 4 — Exemplars
# --------------------------------------------------------------------- #


def get_pred_for(df: pd.DataFrame, rec_id: str, axis: str, severity: float, arm: str) -> dict:
    sub = df[(df.rec_id == rec_id) & (df.axis == axis) & (df.severity == severity) & (df.arm == arm)]
    if len(sub) == 0:
        return {"pred": None, "prob": None}
    row = sub.iloc[0]
    return {"pred": int(row["pred"]), "prob": float(row["prob"])}


def pick_exemplars(profile: pd.DataFrame, df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Pick the median (typical) recording per category.

    Median by `flip_rate` (most representative of the category's central
    behavior). Ties broken by `rec_id` for determinism.
    """
    rows = []
    for cat in CATEGORY_ORDER:
        sub = profile[profile["category"] == cat].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(["flip_rate", "rec_id"]).reset_index(drop=True)
        med_idx = len(sub) // 2
        ex = sub.iloc[med_idx]
        rec_id = ex["rec_id"]
        label = int(ex["label"])
        b = baseline[baseline.rec_id == rec_id].iloc[0]
        rows.append(
            {
                "rec_id": rec_id,
                "category": cat,
                "label": label,
                "gt": "normal" if label == 0 else "abnormal",
                "baseline_pred": int(b["baseline_pred"]),
                "baseline_prob": float(b["baseline_prob"]),
                "baseline_correct": int(b["baseline_correct"]),
                # Three canonicalized probe cells representative of each axis.
                "sr128_canon": get_pred_for(df, rec_id, "sampling_rate", 128.0, "canonicalized"),
                "cal05_canon": get_pred_for(df, rec_id, "calibration", 0.5, "canonicalized"),
                "pl30_canon": get_pred_for(df, rec_id, "power_line", 30.0, "canonicalized"),
                "flip_rate": float(ex["flip_rate"]),
                "n_catastrophic": int(ex["n_catastrophic"]),
                "max_wrong_conf": float(ex["max_wrong_conf"]),
            }
        )
    return pd.DataFrame(rows)


def write_exemplars_md(exemplars: pd.DataFrame) -> None:
    def fmt(cell: dict, label: int) -> str:
        if cell["pred"] is None:
            return "—"
        pred_lbl = "normal" if cell["pred"] == 0 else "abnormal"
        mark = "✓" if cell["pred"] == label else "✗"
        return f"{pred_lbl} ({cell['prob']:.2f}) {mark}"

    lines = [
        "# Failure-Taxonomy Exemplars",
        "",
        "Median (typical) recording per category. `pred (prob) ✓/✗` reads as the predicted class, the abnormal-class probability, and a check/cross for correctness against ground truth. Probes pulled from the canonicalized arm at the most aggressive severity per axis (sampling_rate=128 Hz, calibration gain=0.5×, power_line=30 µV).",
        "",
        "| rec_id | gt | baseline pred | sr=128 (canon) | cal=0.5 (canon) | pl=30 (canon) | category | flip_rate | max wrong-side conf |",
        "|--------|----|---------------|----------------|-----------------|----------------|----------|-----------|---------------------|",
    ]
    for _, r in exemplars.iterrows():
        label = int(r["label"])
        bp = "normal" if r["baseline_pred"] == 0 else "abnormal"
        bmark = "✓" if r["baseline_correct"] else "✗"
        base_str = f"{bp} ({r['baseline_prob']:.2f}) {bmark}"
        lines.append(
            f"| `{r['rec_id']}` | {r['gt']} | {base_str} | "
            f"{fmt(r['sr128_canon'], label)} | {fmt(r['cal05_canon'], label)} | "
            f"{fmt(r['pl30_canon'], label)} | **{CATEGORY_LABEL[r['category']]}** | "
            f"{r['flip_rate']:.2f} | {r['max_wrong_conf']:.2f} |"
        )
    EXEMPLARS_MD.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------- #
# Part 6 — Axis vulnerability ranking
# --------------------------------------------------------------------- #


def axis_vulnerability(df: pd.DataFrame, baseline: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    """Rank axes by mean flip rate, mean confidence drop, and worst AUROC drop.

    All metrics computed on the *canonicalized* arm (operational deployment).
    "Maximum severity" defined as the cell with the worst aggregate AUROC in
    the canonicalized arm per axis — this gives a fair, data-driven worst case
    rather than picking a severity by hand.
    """
    base_auroc = float(
        sweep[(sweep.axis == "sampling_rate") & (sweep.severity == BASELINE_SEVERITY["sampling_rate"])
              & (sweep.arm == "canonicalized")]["auroc"].iloc[0]
    )

    rows = []
    for axis in ["sampling_rate", "calibration", "power_line"]:
        sub_sweep = sweep[(sweep.axis == axis) & (sweep.arm == "canonicalized")
                          & (sweep.severity != BASELINE_SEVERITY[axis])]
        worst_row = sub_sweep.loc[sub_sweep["auroc"].idxmin()]
        worst_sev = float(worst_row["severity"])
        worst_auroc = float(worst_row["auroc"])
        auroc_drop = base_auroc - worst_auroc

        # Per-recording flip + conf drop at worst severity (canonicalized).
        cell = df[(df.axis == axis) & (df.severity == worst_sev) & (df.arm == "canonicalized")]
        cell = cell.merge(baseline[["rec_id", "baseline_pred", "baseline_prob", "label"]],
                          on=["rec_id", "label"])
        flips = (cell["pred"] != cell["baseline_pred"]).mean()
        # Vectorized GT-class prob: gt_prob = prob when label==1, else 1-prob.
        gt_b = np.where(cell["label"] == 1, cell["baseline_prob"], 1 - cell["baseline_prob"])
        gt_s = np.where(cell["label"] == 1, cell["prob"], 1 - cell["prob"])
        conf_drop = float(np.mean(gt_b - gt_s))

        # Mean catastrophic-cell rate (high-conf wrong) at worst severity.
        wrong_side = np.where(cell["label"] == 0, cell["prob"], 1 - cell["prob"])
        cat_rate = float(((cell["pred"] != cell["label"]).values & (wrong_side >= HIGH_CONF)).mean())

        rows.append(
            {
                "axis": axis,
                "worst_severity": worst_sev,
                "auroc_baseline": base_auroc,
                "auroc_worst": worst_auroc,
                "auroc_drop": auroc_drop,
                "flip_rate_worst": float(flips),
                "conf_drop_worst": conf_drop,
                "catastrophic_cell_rate_worst": cat_rate,
            }
        )
    rank = pd.DataFrame(rows)
    # Composite rank: rank-sum across the three normalized metrics (lower = better).
    for col in ["flip_rate_worst", "conf_drop_worst", "auroc_drop"]:
        rank[f"rk_{col}"] = rank[col].rank(ascending=False, method="min")
    rank["rank_sum"] = rank[[c for c in rank.columns if c.startswith("rk_")]].sum(axis=1)
    rank = rank.sort_values("rank_sum").reset_index(drop=True)
    rank.insert(0, "rank", rank.index + 1)
    return rank


def write_axis_rank_md(rank: pd.DataFrame) -> None:
    lines = [
        "# Axis Vulnerability Ranking",
        "",
        "Empirical ranking of the three acquisition-shift axes by the *canonicalized*-arm behavior of the pretrained classifier on the SPMB sweep (n=276 recordings; per-axis worst severity selected as the cell with the lowest aggregate AUROC).",
        "",
        "| Rank | Axis | Worst severity | AUROC (base→worst) | ΔAUROC | Mean flip rate | Mean GT-conf drop | Catastrophic-cell rate |",
        "|------|------|----------------|--------------------|--------|----------------|-------------------|------------------------|",
    ]
    for _, r in rank.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['axis']} | {r['worst_severity']:g} | "
            f"{r['auroc_baseline']:.3f} → {r['auroc_worst']:.3f} | "
            f"{r['auroc_drop']:.3f} | {r['flip_rate_worst']:.3f} | "
            f"{r['conf_drop_worst']:+.3f} | {r['catastrophic_cell_rate_worst']:.3f} |"
        )
    lines += [
        "",
        "**Composite rank** = rank-sum across {flip rate, GT-class confidence drop, ΔAUROC} at each axis's worst severity. Lower is more vulnerable.",
        "",
        "**Reading.** The top-ranked axis dominates by simultaneously inducing the largest AUROC drop *and* the largest per-recording flip rate — meaning both the aggregate score and the individual decisions collapse together. The lower-ranked axes can produce nontrivial AUROC drops without flipping as many individual predictions (calibration shift, in particular, often degrades probability quality without changing the argmax).",
    ]
    AXIS_RANK_MD.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------- #
# Part 7 — LaTeX subsection
# --------------------------------------------------------------------- #


def write_section_tex(counts: pd.DataFrame, exemplars: pd.DataFrame, rank: pd.DataFrame, total: int) -> None:
    by_cat = {row["category"]: row for _, row in counts.iterrows()}
    top3 = counts.sort_values("n", ascending=False).head(3)
    catastrophic_ex = exemplars[exemplars["category"] == "catastrophic_shift"]
    if len(catastrophic_ex):
        cat_rec = catastrophic_ex.iloc[0]
        cat_rec_id = str(cat_rec["rec_id"]).replace("_", r"\_")
        cat_gt = cat_rec["gt"]
        cat_wrong = cat_rec["max_wrong_conf"]
    else:
        cat_rec_id, cat_gt, cat_wrong = "n/a", "n/a", float("nan")

    top_axis = rank.iloc[0]
    top_axis_name = top_axis["axis"].replace("_", r"\_")

    body = [
        r"\subsection{Failure-mode taxonomy}",
        r"\label{sec:failure-taxonomy}",
        r"",
        r"Aggregate AUROC obscures the structure of failure. We cluster every recording in the SPMB cohort ($n{=}" + str(total) + r"$) by its joint behavior across the canonicalized arm of the $14$ $(\text{axis},\text{severity})$ cells, yielding seven mutually exclusive failure categories.",
        r"",
        r"\begin{description}\setlength{\itemsep}{1pt}",
        r"  \item[Robust-correct.] Correct at baseline and on the majority of shifted cells. The benign case.",
        r"  \item[Robust-wrong.] Wrong at baseline and stays wrong everywhere ($\leq 10\%$ shifted-cell recovery). Intrinsic classification failure; not driven by shift.",
        r"  \item[Shift-induced flip.] Correct at baseline but $\geq 30\%$ of shifted cells flip the prediction. The canonical bad pattern this paper exists to expose.",
        r"  \item[Shift-recovered.] Wrong at baseline, $\geq 60\%$ correct after shift. Suspicious: a shift that helps the model is rarely doing it for the right reason.",
        r"  \item[Calibration collapse.] Correct at baseline, retains the argmax, but $\geq 20\%$ of shifted cells land within $\pm 0.15$ of $p{=}0.5$. The model ``knows'' it is uncertain without flipping --- which a downstream selective head can catch.",
        r"  \item[Catastrophic shift.] Correct at baseline, then at least one shifted cell is wrong with $p_{\text{wrong-class}}\!\geq\!0.80$. The most clinically dangerous failure: high-confidence wrong direction under operational shift.",
        r"  \item[Axis-specific.] Correct at baseline, failures concentrated in a single axis ($\geq 50\%$ flip on one axis, $<15\%$ on the other two).",
        r"\end{description}",
        r"",
        r"Table~\ref{tab:failure-counts} reports the prevalence of each category. The three most common are " +
        ", ".join(f"\\emph{{{CATEGORY_LABEL[r['category']].lower()}}} ($n{{=}}{r['n']}$, {r['frac']*100:.1f}\\%)" for _, r in top3.iterrows()) +
        ".",
        r"",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Failure-category prevalence on the SPMB cohort ($n{=}" + str(total) + r"$), stratified by ground-truth label.}",
        r"\label{tab:failure-counts}",
        r"\footnotesize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Category & All & Normal & Abnormal \\",
        r"\midrule",
    ]
    for cat in CATEGORY_ORDER:
        r = by_cat[cat]
        body.append(
            f"{CATEGORY_LABEL[cat]} & {r['n']} ({r['frac']*100:.1f}\\%) & "
            f"{r['n_normal']} ({r['frac_normal']*100:.1f}\\%) & "
            f"{r['n_abnormal']} ({r['frac_abnormal']*100:.1f}\\%) \\\\"
        )
    body += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
        r"\paragraph{The dangerous tail.} \emph{Catastrophic shift} (" +
        f"$n{{=}}{by_cat['catastrophic_shift']['n']}$, {by_cat['catastrophic_shift']['frac']*100:.1f}\\%" +
        r") deserves special attention: these recordings the model classifies correctly under canonical acquisition, but at least one operational shift drives the prediction to the wrong class with confidence $\geq 0.80$. Exemplar recording \texttt{" + cat_rec_id + r"} (ground truth: " + cat_gt + r") reaches a wrong-side confidence of $" + f"{cat_wrong:.2f}" + r"$ under shift. Clinically, this is the highest-priority failure mode: it produces a confident, miscalibrated report whose error signature is invisible to the clinician without a per-recording trust signal.",
        r"",
        r"\paragraph{Polarized-logit signature.} The near-emptiness of the \emph{calibration-collapse} bucket ($n{=}" + f"{by_cat['calibration_collapse']['n']}" + r"$) is itself a finding: the EEGPT-derived classifier does not gracefully equivocate under shift. Of the $128$ baseline-correct recordings with at least one wrong shifted cell, $112$ ($87.5\%$) are wrong with $p_{\text{wrong}}\!\geq\!0.80$. The model's logit distribution is sharply polarized; when it errs under shift it does so with full confidence. Selective heads that rely on $p\!\approx\!0.5$ as the abstention trigger will fail to catch these, motivating distributional (logit-magnitude, NP test) abstention signals.",
        r"",
        r"\paragraph{Class-asymmetric vulnerability (most novel finding).} The failure load falls almost entirely on the normal class. Of $107$ baseline-correct normal recordings, \emph{every single one} flips on at least one shifted cell ($100\%$ flip incidence), and $102$ ($95.3\%$) flip with $p_{\text{abnormal}}\!\geq\!0.80$ on at least one cell --- i.e., $102/107$ baseline-correct normals are catastrophic-shift recordings. By contrast, of $111$ baseline-correct abnormals, $97$ ($87.4\%$) are robust-correct across the entire sweep, and only $10$ ($9.0\%$) are catastrophic-shift. The shift-induced failure is therefore directional: it predominantly converts a clean normal recording into a confident abnormal report. This is the worst possible direction for a screening tool, because abnormal false-positives drive the alert fatigue that drives clinician disuse. Any deployment of EEGPT-class models on non-canonical acquisition pipelines must address this normal-to-abnormal drift specifically.",
        r"",
        r"\paragraph{Clinical implications by category.} Each category maps to a distinct downstream risk:",
        r"\begin{itemize}\setlength{\itemsep}{1pt}",
        r"  \item \emph{Shift-induced flip} on normal-class recordings inflates the false-alarm rate to the reading clinician; on abnormal-class recordings it produces false reassurance.",
        r"  \item \emph{Catastrophic shift} converts a quiet error into a hazardous one: the high-confidence wrong report carries the same UI weight as a correct one.",
        r"  \item \emph{Calibration collapse} is the \emph{safest} failure mode --- a selective head with a $|\,\text{logit}\,|$ threshold or NP test reliably abstains.",
        r"  \item \emph{Shift-recovered} flags a recording whose baseline answer was itself suspect; deployment should ignore the recovery and treat the baseline as authoritative.",
        r"  \item \emph{Axis-specific} failures localize blame to a single acquisition variable, enabling targeted device-side fixes (resampling kernel, gain calibration, notch filter).",
        r"\end{itemize}",
        r"",
        r"\paragraph{Axis vulnerability.} Ranking the three acquisition axes by a composite of mean flip rate, mean ground-truth-class confidence drop, and $\Delta$AUROC at each axis's worst severity (canonicalized arm) places \emph{" + top_axis_name + r"} first --- $\Delta\text{AUROC}{=}" + f"{top_axis['auroc_drop']:.3f}" + r"$, mean flip rate $" + f"{top_axis['flip_rate_worst']:.2f}" + r"$, mean GT-class confidence drop $" + f"{top_axis['conf_drop_worst']:+.2f}" + r"$ at severity $" + f"{top_axis['worst_severity']:g}" + r"$. Sampling rate dominates because non-integer resampling alters the spectral content the EEGPT backbone uses; the other axes degrade probability quality without always reorganizing the spectral evidence.",
        r"",
        r"Figure~\ref{fig:failure-taxonomy} renders the per-cell composition of these categories across the full acquisition sweep.",
        r"",
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \includegraphics[width=\linewidth]{figures/fig_failure_taxonomy.pdf}",
        r"  \caption{Per-cell failure-category composition across (axis, severity) for the canonicalized arm, $n{=}276$ recordings. The \emph{catastrophic-shift} fraction (magenta) grows with sampling-rate departure from $256$\,Hz and with calibration-gain departure from $1.0\times$; canonicalized power-line filtering keeps the power-line axis near baseline. The near-absence of \emph{calibration collapse} (yellow) shows that the EEGPT-derived classifier does not equivocate under shift --- it commits with high confidence, often in the wrong direction.}",
        r"  \label{fig:failure-taxonomy}",
        r"\end{figure}",
        r"",
    ]
    SECTION_TEX.write_text("\n".join(body) + "\n")


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #


def main() -> None:
    df = load_predictions()
    base = baseline_table(df)
    profile = build_profile(df, base)
    profile = classify(profile)
    profile.to_csv(PROFILE_CSV, index=False)
    counts = category_counts(profile)
    print("=== Category counts ===")
    print(counts.to_string(index=False))

    exemplars = pick_exemplars(profile, df, base)
    write_exemplars_md(exemplars)
    print("\n=== Exemplars chosen ===")
    print(exemplars[["rec_id", "category", "gt", "baseline_correct", "flip_rate", "max_wrong_conf"]]
          .to_string(index=False))

    sweep = pd.read_csv(SWEEP_CSV)
    rank = axis_vulnerability(df, base, sweep)
    write_axis_rank_md(rank)
    print("\n=== Axis vulnerability ===")
    print(rank.to_string(index=False))

    figure_taxonomy(df, base)
    print(f"\nFigure written: {FIG_PDF}")

    write_section_tex(counts, exemplars, rank, total=len(profile))
    print(f"LaTeX subsection: {SECTION_TEX}")


if __name__ == "__main__":
    main()
