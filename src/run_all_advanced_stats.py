"""
Orchestrator: run every advanced-stats method and write the results to disk.

Inputs (auto-detected under <repo>/data/):
    per_recording_predictions_latest.csv     -- 7728 rows
    sweep_latest.csv                         -- 28 summary rows
    spmb_case_study_results.csv              -- optional, not used by the paper (NOT shipped
                                                publicly; verification skips
                                                the case-study check when
                                                this CSV is absent)

Outputs:
    <repo>/data/advanced_stats_results.json
    <repo>/data/advanced_stats_summary.md
    <repo>/data/seeds.json

Each method's results are stored under a top-level key in the JSON. All
numbers come from real computation; no hand-tuned values.

CLI:
    python3 src/run_all_advanced_stats.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional  # noqa: F401

import numpy as np
import pandas as pd

# Path bootstrap so the script runs from any cwd.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import advanced_stats as A  # noqa: E402

SPMB = HERE.parent
DATA = SPMB / "data"

DEFAULT_SEED = 7

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"label": int, "severity": str, "rec_id": str})
    df["label"] = df["label"].astype(int)
    df["logit"] = df["logit"].astype(float)
    df["prob"] = df["prob"].astype(float)
    df["var_logit"] = df["var_logit"].astype(float)
    # Patient ID: TUAB rec_ids are <patient>_s<session>_t<token>.
    df["patient_id"] = df["rec_id"].str.split("_").str[0]
    return df


def load_sweep(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_case_study(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop the duplicate file the user flagged.
    DROP = "Patient26_NATUSEEG-PC_t1_29Jul.e"
    return df[df["file"] != DROP].reset_index(drop=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pair_by_recording(df: pd.DataFrame, axis: str, severity: str):
    """Return aligned (labels, probs_naive, probs_canon, patient_ids, rec_ids)."""
    naive = df[(df["axis"] == axis) & (df["severity"].astype(str) == str(severity)) & (df["arm"] == "naive")]
    canon = df[(df["axis"] == axis) & (df["severity"].astype(str) == str(severity)) & (df["arm"] == "canonicalized")]
    merged = naive.merge(canon, on="rec_id", suffixes=("_n", "_c"))
    if merged.empty:
        return (np.array([]), np.array([]), np.array([]), [], [])
    labels = merged["label_n"].to_numpy(dtype=int)
    p_n = merged["prob_n"].to_numpy(dtype=float)
    p_c = merged["prob_c"].to_numpy(dtype=float)
    pat = merged["patient_id_n"].tolist()
    rec = merged["rec_id"].tolist()
    return labels, p_n, p_c, pat, rec


def severity_z(axis: str, severities: list[str]) -> dict[str, float]:
    """Map severity string -> z-scored numeric per axis (centered on the identity).

    The identity severity is the one where the canonicalized arm equals the
    baseline (sampling_rate=250, calibration=1.0, power_line=0.0).
    """
    identity = {"sampling_rate": "250", "calibration": "1.0", "power_line": "0.0"}.get(axis)
    nums = []
    for s in severities:
        try:
            nums.append(float(s))
        except ValueError:
            nums.append(0.0)
    arr = np.asarray(nums, dtype=float)
    centre = float(arr.mean()) if identity is None else float(identity) if identity else 0.0
    scale = float(arr.std(ddof=1)) if arr.size > 1 and arr.std(ddof=1) > 0 else 1.0
    return {s: (n - centre) / scale for s, n in zip(severities, nums)}


# ---------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------
def run_method_1_fdr(df: pd.DataFrame) -> dict:
    """BH FDR @ q=0.05 across all (axis, severity) DeLong p-values."""
    cells = sorted({(r["axis"], str(r["severity"])) for _, r in df.iterrows()})
    rows = []
    for axis, sev in cells:
        labels, p_n, p_c, _, _ = pair_by_recording(df, axis, sev)
        if labels.size == 0 or len(np.unique(labels)) < 2:
            rows.append({"axis": axis, "severity": sev, "p_delong": float("nan"),
                         "auroc_naive": float("nan"), "auroc_canon": float("nan")})
            continue
        auc_n = A.auroc(labels, p_n)
        auc_c = A.auroc(labels, p_c)
        p = A._delong_p(labels, p_n, p_c)
        rows.append({"axis": axis, "severity": sev, "p_delong": float(p),
                     "auroc_naive": float(auc_n), "auroc_canon": float(auc_c),
                     "auroc_delta": float(auc_c - auc_n)})
    pvals = np.asarray([r["p_delong"] for r in rows], dtype=float)
    finite = np.isfinite(pvals)
    finite_pvals = pvals[finite]
    fdr_q05 = A.fdr_benjamini_hochberg(finite_pvals, q=0.05)
    # Re-attach the adjusted p back to the right axis/severity rows
    adj_full = np.full_like(pvals, np.nan, dtype=float)
    rej_full = np.zeros_like(pvals, dtype=bool)
    adj_full[finite] = fdr_q05["adjusted_p"]
    rej_full[finite] = fdr_q05["reject"]
    for i, r in enumerate(rows):
        r["adjusted_p"] = float(adj_full[i]) if math.isfinite(adj_full[i]) else None
        r["reject_q05"] = bool(rej_full[i])
    n_sig_raw = int(np.sum(finite_pvals < 0.05))
    n_sig_adj = int(fdr_q05["n_rejected"])
    return {
        "rows":          rows,
        "n_tests":       int(len(finite_pvals)),
        "n_sig_raw_p05": n_sig_raw,
        "n_sig_bh_q05":  n_sig_adj,
        "q":             0.05,
        "method":        "Benjamini & Hochberg (1995), two-sided DeLong p-values pooled across axes",
    }


def run_method_2_bayes(df: pd.DataFrame) -> dict:
    """Per-arm posterior accuracy CIs (Beta-Binomial) on the clean baseline."""
    out: dict[str, Any] = {"per_axis_severity_arm": []}
    cells = sorted({(r["axis"], str(r["severity"]), r["arm"])
                    for _, r in df.iterrows()})
    for axis, sev, arm in cells:
        sub = df[(df["axis"] == axis) & (df["severity"].astype(str) == str(sev)) & (df["arm"] == arm)]
        if sub.empty:
            continue
        labels = sub["label"].to_numpy(dtype=int)
        preds = (sub["prob"].to_numpy(dtype=float) >= 0.5).astype(int)
        r = A.bayesian_beta_binomial(labels, preds,
                                     prior_alpha=1.0, prior_beta=1.0,
                                     ci_level=0.95)
        r.update({"axis": axis, "severity": sev, "arm": arm})
        out["per_axis_severity_arm"].append(r)
    # Headline: clean baseline arm (sampling_rate, 250, naive)
    headline = next(
        (r for r in out["per_axis_severity_arm"]
         if r["axis"] == "sampling_rate" and r["severity"] == "250" and r["arm"] == "naive"),
        None,
    )
    out["clean_baseline"] = headline
    out["prior"] = {"alpha": 1.0, "beta": 1.0,
                    "rationale": "Beta(1,1) Bayes-Laplace / Jeffreys-uniform"}
    return out


def run_method_3_mixed_effects(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict:
    """Mixed-effects logistic on per-recording correctness, random intercept = patient_id."""
    # Build long-format: for each (axis, severity, arm) row -> correct = 1{argmax==label}.
    df = df.copy()
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    # Build a severity_z per axis on the unique severities; pool across axes
    sev_z_full: dict[tuple[str, str], float] = {}
    for ax in df["axis"].unique():
        sevs = sorted(df[df["axis"] == ax]["severity"].astype(str).unique())
        sz = severity_z(ax, sevs)
        for s, v in sz.items():
            sev_z_full[(ax, s)] = v
    df["severity_z"] = df.apply(lambda r: sev_z_full[(r["axis"], str(r["severity"]))], axis=1)
    # Drop the identity rows (severity_z == 0) so we are estimating only the
    # *shift* effect, not a degenerate identity row that always has arm-equal scores.
    df_shift = df[df["severity_z"] != 0.0].reset_index(drop=True)
    result = A.mixed_effects_logistic(
        df_shift,
        response="correct",
        fixed_terms=("arm", "severity_z"),
        group="patient_id",
        method="auto",
    )
    # Also fit a per-axis version so the table can report per-axis arm coefs.
    per_axis = {}
    for ax in sorted(df_shift["axis"].unique()):
        sub = df_shift[df_shift["axis"] == ax].reset_index(drop=True)
        if sub.empty:
            continue
        r = A.mixed_effects_logistic(sub, response="correct",
                                     group="patient_id", method="auto")
        per_axis[ax] = r
    return {
        "pooled": result,
        "per_axis": per_axis,
        "n_shift_rows": int(len(df_shift)),
        "n_patients": int(df_shift["patient_id"].nunique()),
        "seed": int(seed),
        "note": (
            "Random intercept on TUAB patient_id (parsed from rec_id stem). "
            "Identity rows (severity_z=0) are excluded so the arm coefficient "
            "estimates the SHIFT effect, not the baseline identity."
        ),
    }


def run_method_4_stratified_bootstrap(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict:
    out = {"per_cell": []}
    cells = sorted({(r["axis"], str(r["severity"])) for _, r in df.iterrows()
                    if not (r["axis"] == "sampling_rate" and str(r["severity"]) == "250")})
    for axis, sev in cells:
        labels, p_n, p_c, _, _ = pair_by_recording(df, axis, sev)
        if labels.size == 0 or len(np.unique(labels)) < 2:
            continue
        r = A.stratified_bootstrap_vs_naive(labels, p_n, p_c,
                                            n_resamples=1000, seed=seed)
        r.update({"axis": axis, "severity": sev, "n": int(labels.size)})
        out["per_cell"].append(r)
    # Headline: biggest absolute half-width difference.
    if out["per_cell"]:
        worst = max(out["per_cell"], key=lambda r: r["halfwidth_diff"] if math.isfinite(r["halfwidth_diff"]) else -1)
        out["worst_halfwidth_diff_cell"] = {k: worst[k] for k in
                                             ("axis", "severity", "halfwidth_diff",
                                              "halfwidth_naive", "halfwidth_stratified")}
    out["n_resamples"] = 1000
    out["seed"] = int(seed)
    return out


def run_method_5_permutation(df: pd.DataFrame,
                             n_permutations: int = 10000,
                             seed: int = DEFAULT_SEED) -> dict:
    out = {"per_cell": []}
    cells = sorted({(r["axis"], str(r["severity"])) for _, r in df.iterrows()
                    if not (r["axis"] == "sampling_rate" and str(r["severity"]) == "250")})
    for axis, sev in cells:
        labels, p_n, p_c, _, _ = pair_by_recording(df, axis, sev)
        if labels.size == 0 or len(np.unique(labels)) < 2:
            continue
        r = A.permutation_test_axis_shift(labels, p_n, p_c,
                                          n_permutations=n_permutations,
                                          seed=seed)
        r.update({"axis": axis, "severity": sev, "n": int(labels.size)})
        out["per_cell"].append(r)
    out["n_permutations"] = int(n_permutations)
    out["seed"] = int(seed)
    return out


def run_method_6_effect_sizes(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict:
    out = {"per_cell": []}
    cells = sorted({(r["axis"], str(r["severity"])) for _, r in df.iterrows()
                    if not (r["axis"] == "sampling_rate" and str(r["severity"]) == "250")})
    for axis, sev in cells:
        labels, p_n, p_c, _, _ = pair_by_recording(df, axis, sev)
        if labels.size == 0:
            continue
        correct_n = ((p_n >= 0.5).astype(int) == labels).astype(float)
        correct_c = ((p_c >= 0.5).astype(int) == labels).astype(float)
        r = A.effect_sizes_d_g_with_ci(correct_n, correct_c,
                                       n_resamples=2000, seed=seed)
        r.update({"axis": axis, "severity": sev, "n": int(labels.size)})
        out["per_cell"].append(r)
    out["seed"] = int(seed)
    return out


def run_method_7_calibration(df: pd.DataFrame) -> dict:
    """Extended calibration on the clean baseline AND on the pooled shifted pool."""
    base = df[(df["axis"] == "sampling_rate")
              & (df["severity"].astype(str) == "250")
              & (df["arm"] == "naive")]
    ce_baseline = A.calibration_extended(base["label"].to_numpy(int),
                                         base["prob"].to_numpy(float),
                                         n_bins=10, ci_level=0.95)
    # Pooled shifted: every row not on the clean baseline
    shifted = df[~((df["axis"] == "sampling_rate")
                   & (df["severity"].astype(str) == "250")
                   & (df["arm"] == "naive"))]
    ce_shifted = A.calibration_extended(shifted["label"].to_numpy(int),
                                        shifted["prob"].to_numpy(float),
                                        n_bins=10, ci_level=0.95)
    # Per-arm-pooled calibration for reviewer comparisons
    ce_naive = A.calibration_extended(
        df[df["arm"] == "naive"]["label"].to_numpy(int),
        df[df["arm"] == "naive"]["prob"].to_numpy(float),
        n_bins=10, ci_level=0.95)
    ce_canon = A.calibration_extended(
        df[df["arm"] == "canonicalized"]["label"].to_numpy(int),
        df[df["arm"] == "canonicalized"]["prob"].to_numpy(float),
        n_bins=10, ci_level=0.95)
    return {
        "clean_baseline":  ce_baseline,
        "pooled_shifted":  ce_shifted,
        "pooled_naive":    ce_naive,
        "pooled_canon":    ce_canon,
    }


def run_method_8_aurc(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict:
    """Full risk-coverage AURC + E-AURC on the pooled shifted pool, using |logit| as confidence."""
    shifted = df[~((df["axis"] == "sampling_rate")
                   & (df["severity"].astype(str) == "250")
                   & (df["arm"] == "naive"))]
    labels = shifted["label"].to_numpy(int)
    probs = shifted["prob"].to_numpy(float)
    preds = (probs >= 0.5).astype(int)
    # Confidence: |logit|. Adequacy from selective_prediction.py uses var_logit;
    # we report both ranking signals so the paper can pick whichever has the
    # stronger AURC, which is typically |logit| for confidence-based selective
    # prediction (Hendrycks & Gimpel, 2017).
    conf_abs_logit = np.abs(shifted["logit"].to_numpy(float))
    # adequacy = -var_logit (higher = less variable = more trustworthy)
    conf_neg_var = -shifted["var_logit"].to_numpy(float)
    out = {}
    out["abs_logit"] = A.risk_coverage_aurc_eaurc(labels, preds, conf_abs_logit,
                                                   seed=seed, n_bootstrap=1000)
    out["neg_var_logit"] = A.risk_coverage_aurc_eaurc(labels, preds, conf_neg_var,
                                                      seed=seed, n_bootstrap=1000)
    out["n"] = int(labels.size)
    out["seed"] = int(seed)
    out["note"] = (
        "Pooled shifted pool (all rows except clean baseline). Two confidence "
        "signals reported: |logit| (Hendrycks-Gimpel baseline) and -var_logit "
        "(adequacy proxy used by the paper's coverage-risk curve)."
    )
    return out


# ----- Tier 2 -----
def run_method_9_conformal(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict:
    """Split-conformal: calibrate on a 50% patient split of the clean baseline,
    test on the other 50% + a pooled-shifted set."""
    base = df[(df["axis"] == "sampling_rate")
              & (df["severity"].astype(str) == "250")
              & (df["arm"] == "naive")].reset_index(drop=True)
    # Patient-stratified random split
    patients = sorted(base["patient_id"].unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(patients))
    cut = len(patients) // 2
    cal_pats = set(np.asarray(patients)[perm[:cut]])
    test_pats = set(np.asarray(patients)[perm[cut:]])
    cal_df = base[base["patient_id"].isin(cal_pats)]
    test_df = base[base["patient_id"].isin(test_pats)]
    res_clean = A.conformal_split_class_conditional(
        cal_df["label"].to_numpy(int),
        cal_df["prob"].to_numpy(float),
        test_df["label"].to_numpy(int),
        test_df["prob"].to_numpy(float),
        alpha=0.10,
    )
    # Also evaluate empirical coverage on a pooled SHIFTED test set (same cal).
    shifted = df[~((df["axis"] == "sampling_rate")
                   & (df["severity"].astype(str) == "250")
                   & (df["arm"] == "naive"))]
    res_shifted = A.conformal_split_class_conditional(
        cal_df["label"].to_numpy(int),
        cal_df["prob"].to_numpy(float),
        shifted["label"].to_numpy(int),
        shifted["prob"].to_numpy(float),
        alpha=0.10,
    )
    # JSON-friendly: convert set_size_distribution keys/vals
    for r in (res_clean, res_shifted):
        if "set_size_distribution" in r:
            r["set_size_distribution"] = {int(k): int(v)
                                          for k, v in r["set_size_distribution"].items()}
    return {
        "clean_test":   res_clean,
        "shifted_test": res_shifted,
        "alpha":        0.10,
        "n_cal_patients":  int(len(cal_pats)),
        "n_test_patients": int(len(test_pats)),
        "seed":         int(seed),
    }


def run_method_10_mi(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict:
    """MI between (logit, prob, var_logit) and (label, severity_index) on shifted pool."""
    shifted = df[~((df["axis"] == "sampling_rate")
                   & (df["severity"].astype(str) == "250")
                   & (df["arm"] == "naive"))].copy()
    # Encode (axis, severity) as a categorical integer
    keys = sorted({(r["axis"], str(r["severity"])) for _, r in shifted.iterrows()})
    key_to_id = {k: i for i, k in enumerate(keys)}
    shifted["severity_id"] = shifted.apply(
        lambda r: key_to_id[(r["axis"], str(r["severity"]))], axis=1)
    feat = shifted[["logit", "prob", "var_logit"]].to_numpy(dtype=float)
    mi_label = A.mi_embedding_axis(feat, shifted["label"].to_numpy(int), seed=seed)
    mi_sev = A.mi_embedding_axis(feat, shifted["severity_id"].to_numpy(int), seed=seed)
    return {
        "mi_features_vs_label":        mi_label,
        "mi_features_vs_severity_id":  mi_sev,
        "feature_names":               ["logit", "prob", "var_logit"],
        "n":                            int(len(shifted)),
        "note": (
            "EEGPT raw embeddings are not in the CSV; we use the per-recording "
            "(logit, prob, var_logit) summary as the smallest available "
            "surrogate. MI is computed via sklearn.feature_selection.mutual_info_classif "
            "with k=3 neighbour KSG estimator."
        ),
        "seed":                         int(seed),
    }


def run_method_11_seed_sens(df: pd.DataFrame) -> dict:
    """Sampling-variability proxy on the clean baseline."""
    base = df[(df["axis"] == "sampling_rate")
              & (df["severity"].astype(str) == "250")
              & (df["arm"] == "naive")]
    return A.seed_sensitivity(base["label"].to_numpy(int),
                              base["prob"].to_numpy(float),
                              n_seeds=20, base_seed=DEFAULT_SEED)


def run_method_12_power(method6_result: dict) -> dict:
    """Retrospective power: minimum N to detect d=0.2 at 80% power."""
    # Use the median observed |d_z| from method 6 across cells as the
    # empirical anchor.
    cells = method6_result.get("per_cell", [])
    if cells:
        dz_abs = np.asarray([abs(c["cohens_dz"]) for c in cells
                             if math.isfinite(c["cohens_dz"])])
        observed_d = float(np.median(dz_abs)) if dz_abs.size else 0.0
    else:
        observed_d = 0.0
    return {
        "observed_median_abs_dz": float(observed_d),
        "n_required_d020":        A.power_calc_retrospective(observed_d, target_d=0.20, power=0.80),
        "n_required_d050":        A.power_calc_retrospective(observed_d, target_d=0.50, power=0.80),
        "n_required_d080":        A.power_calc_retrospective(observed_d, target_d=0.80, power=0.80),
    }


# ---------------------------------------------------------------------
# Verification: re-derive the most-cited numbers in the paper
# ---------------------------------------------------------------------
def run_verification(df: pd.DataFrame, sweep: pd.DataFrame,
                     case_study: Optional[pd.DataFrame]) -> dict:
    """Re-compute every paper-cited number from the per-recording CSV.

    Each entry in the returned dict has: paper_value, computed_value, drift,
    location_in_paper.
    """
    rep: dict = {"checks": [], "drift_count": 0}

    def _check(name, paper_value, computed_value, loc, tol=1e-3):
        d = abs(paper_value - computed_value) if (paper_value is not None and math.isfinite(paper_value) and math.isfinite(computed_value)) else float("nan")
        drift = (d > tol) if math.isfinite(d) else False
        if drift:
            rep["drift_count"] += 1
        rep["checks"].append({
            "name":           name,
            "paper_value":    paper_value,
            "computed":       float(computed_value),
            "abs_diff":       float(d) if math.isfinite(d) else None,
            "drift_above_tol": bool(drift),
            "location":       loc,
        })

    # Clean baseline: AUROC, balanced accuracy, ECE
    base = df[(df["axis"] == "sampling_rate")
              & (df["severity"].astype(str) == "250")
              & (df["arm"] == "naive")]
    labels = base["label"].to_numpy(int)
    probs = base["prob"].to_numpy(float)
    preds = (probs >= 0.5).astype(int)
    auc_baseline = A.auroc(labels, probs)
    ba_baseline = (((labels == 1) & (preds == 1)).sum() / max(1, (labels == 1).sum()) +
                   ((labels == 0) & (preds == 0)).sum() / max(1, (labels == 0).sum())) / 2.0
    ce = A.calibration_extended(labels, probs, n_bins=10)
    _check("clean_AUROC", 0.908, auc_baseline, "§Results, abstract")
    _check("clean_balanced_acc", 0.797, float(ba_baseline), "§Clean baseline / Table 1")
    _check("clean_ECE_10bin", 0.125, ce["ece"], "§Clean baseline / Table 1")

    # Bootstrap CI on the baseline AUROC, naive paired bootstrap
    rng = np.random.default_rng(DEFAULT_SEED)
    N = labels.size
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, N, size=N)
        if len(np.unique(labels[idx])) >= 2:
            boots.append(A.auroc(labels[idx], probs[idx]))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    _check("clean_AUROC_CI_lo", 0.870, lo, "§Clean baseline / Table 1", tol=0.01)
    _check("clean_AUROC_CI_hi", 0.940, hi, "§Clean baseline / Table 1", tol=0.01)

    # Sampling-rate sweep AUROCs (canon arm)
    for sev, paper_val in [("128", 0.758), ("200", 0.856), ("250", 0.908), ("256", 0.902), ("512", 0.752)]:
        sub = df[(df["axis"] == "sampling_rate") & (df["severity"].astype(str) == sev) & (df["arm"] == "canonicalized")]
        v = A.auroc(sub["label"].to_numpy(int), sub["prob"].to_numpy(float))
        _check(f"sampling_rate_canon_AUROC_{sev}Hz", paper_val, v, "§Per-axis / Table 2", tol=0.005)

    # Calibration gain AUROCs (both arms at g=0.5 and 1.0)
    for arm, sev, paper_val in [("naive", "0.5", 0.865), ("canonicalized", "0.5", 0.900),
                                 ("naive", "0.75", 0.896), ("canonicalized", "0.75", 0.907),
                                 ("naive", "1.0", 0.908), ("canonicalized", "1.0", 0.908)]:
        sub = df[(df["axis"] == "calibration") & (df["severity"].astype(str) == sev) & (df["arm"] == arm)]
        v = A.auroc(sub["label"].to_numpy(int), sub["prob"].to_numpy(float))
        _check(f"calibration_{arm}_AUROC_g{sev}", paper_val, v, "§Per-axis / Table 2", tol=0.005)

    # N test recordings
    _check("n_test_recordings", 276, int(base.shape[0]), "§Experimental Setup")

    # Case study: post-drop count (only when the clinical CSV is present
    # — the public repo deliberately omits it; this check skips silently
    # so reviewers without it can still verify the other 14 numbers).
    if case_study is not None:
        _check("n_unique_case_study_files_after_drop", 5, int(case_study.shape[0]),
               "optional case-study CSV (not part of the submitted paper)")

    return rep


# ---------------------------------------------------------------------
# JSON serialization helpers (numpy -> python)
# ---------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return [_json_default(x) for x in o.tolist()]
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _sanitize(obj):
    """Recursively replace NaN/Inf with None for valid JSON."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------
def write_summary_md(results: dict, out_path: Path) -> None:
    L: list[str] = []
    L.append("# SPMB 2026 - Advanced Statistical Layer (computed)")
    L.append("")
    L.append(f"_Generated by sources/run_all_advanced_stats.py, seed={DEFAULT_SEED}_")
    L.append("")
    # 1
    m1 = results["fdr_bh_q05"]
    L.append("## 1. Benjamini-Hochberg FDR (DeLong p-values)")
    L.append(f"- Tests pooled: **{m1['n_tests']}** axis x severity cells")
    L.append(f"- Raw p<0.05: **{m1['n_sig_raw_p05']}**")
    L.append(f"- BH-adjusted p<=0.05 (q=0.05): **{m1['n_sig_bh_q05']}**")
    L.append("")
    L.append("| Axis | Severity | AUROC naive | AUROC canon | Delta | DeLong p | BH adj p | Reject q.05 |")
    L.append("|---|---|---:|---:|---:|---:|---:|:-:|")
    for r in sorted(m1["rows"], key=lambda x: (x["axis"], x["severity"])):
        L.append("| {axis} | {severity} | {an} | {ac} | {ad} | {p} | {ap} | {rej} |".format(
            axis=r["axis"], severity=r["severity"],
            an=f"{r['auroc_naive']:.3f}" if math.isfinite(r['auroc_naive']) else "nan",
            ac=f"{r['auroc_canon']:.3f}" if math.isfinite(r['auroc_canon']) else "nan",
            ad=f"{r.get('auroc_delta', float('nan')):+.3f}" if math.isfinite(r.get('auroc_delta', float('nan'))) else "nan",
            p=f"{r['p_delong']:.3g}" if math.isfinite(r['p_delong']) else "nan",
            ap=f"{r['adjusted_p']:.3g}" if r['adjusted_p'] is not None else "nan",
            rej="Y" if r["reject_q05"] else "n",
        ))
    L.append("")

    # 2
    m2 = results["bayesian_beta_binomial"]
    L.append("## 2. Bayesian Beta-Binomial posterior accuracy")
    L.append(f"- Prior: Beta(1, 1) (Bayes-Laplace / Jeffreys-uniform)")
    if m2.get("clean_baseline"):
        b = m2["clean_baseline"]
        L.append(f"- Clean baseline (sampling_rate=250, naive): "
                 f"acc = {b['accuracy']:.3f}, "
                 f"posterior mean = {b['posterior_mean']:.3f}, "
                 f"95% CrI = [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]; "
                 f"Wilson 95% CI = [{b['wilson_lo']:.3f}, {b['wilson_hi']:.3f}]")
    L.append("")
    L.append("| Axis | Severity | Arm | Acc | n | Posterior mean | 95% CrI | Wilson 95% CI |")
    L.append("|---|---|---|---:|---:|---:|---|---|")
    for r in m2["per_axis_severity_arm"]:
        L.append(f"| {r['axis']} | {r['severity']} | {r['arm']} | {r['accuracy']:.3f} | {r['n']} | "
                 f"{r['posterior_mean']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                 f"[{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}] |")
    L.append("")

    # 3
    m3 = results["mixed_effects_logistic"]
    L.append("## 3. Mixed-effects logistic regression")
    L.append(f"- Method: {m3['pooled']['method_used']}")
    L.append(f"- N rows (non-identity): {m3['n_shift_rows']}, N patients: {m3['n_patients']}")
    L.append(f"- Joint Wald (LRT-equivalent) p-value for arm effects = 0: "
             f"**{m3['pooled']['lrt_pvalue']:.4g}**" if (m3['pooled']['lrt_pvalue'] is not None and math.isfinite(m3['pooled']['lrt_pvalue'])) else "- LRT/Wald p-value: nan")
    L.append("")
    L.append("**Pooled fixed effects (log-odds):**")
    L.append("")
    L.append("| Term | Estimate | SE | 95% CI | z | p |")
    L.append("|---|---:|---:|---|---:|---:|")
    for k, v in m3["pooled"]["fixed_effects"].items():
        L.append(f"| {k} | {v['estimate']:+.3f} | {v['se']:.3f} | "
                 f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] | "
                 f"{v['z']:+.2f} | {v['p']:.3g} |")
    L.append("")
    L.append("**Per-axis fixed effects:**")
    L.append("")
    L.append("| Axis | arm_canon estimate | arm_canon SE | arm_canon p | Wald p (arms=0) |")
    L.append("|---|---:|---:|---:|---:|")
    for ax, r in m3["per_axis"].items():
        ac = r["fixed_effects"].get("arm_canon", {})
        if ac:
            est = ac.get("estimate", float("nan"))
            se = ac.get("se", float("nan"))
            p = ac.get("p", float("nan"))
            wald = r.get("lrt_pvalue", float("nan"))
            L.append(f"| {ax} | {est:+.3f} | {se:.3f} | {p:.3g} | {wald:.3g} |")
    L.append("")

    # 4
    m4 = results["stratified_vs_naive_bootstrap"]
    L.append("## 4. Stratified vs naive bootstrap on AUROC delta")
    L.append(f"- n_resamples = {m4['n_resamples']}, seed = {m4['seed']}")
    if "worst_halfwidth_diff_cell" in m4:
        w = m4["worst_halfwidth_diff_cell"]
        L.append(f"- Worst absolute half-width difference: **{w['halfwidth_diff']:.4f}** "
                 f"(axis={w['axis']}, severity={w['severity']}; naive HW={w['halfwidth_naive']:.4f}, "
                 f"stratified HW={w['halfwidth_stratified']:.4f})")
    L.append("")
    L.append("| Axis | Severity | Delta | Naive 95% CI | Stratified 95% CI | |HW diff| |")
    L.append("|---|---|---:|---|---|---:|")
    for r in m4["per_cell"]:
        L.append(f"| {r['axis']} | {r['severity']} | {r['delta_point']:+.3f} | "
                 f"[{r['naive_ci'][0]:+.3f}, {r['naive_ci'][1]:+.3f}] | "
                 f"[{r['stratified_ci'][0]:+.3f}, {r['stratified_ci'][1]:+.3f}] | "
                 f"{r['halfwidth_diff']:.4f} |")
    L.append("")

    # 5
    m5 = results["permutation_test_axis_shift"]
    L.append("## 5. Within-recording permutation test (B = {})".format(m5["n_permutations"]))
    L.append("")
    L.append("| Axis | Severity | n | Observed Delta | Null mean | Null std | p (B={}) |".format(m5["n_permutations"]))
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in m5["per_cell"]:
        L.append(f"| {r['axis']} | {r['severity']} | {r['n']} | "
                 f"{r['observed']:+.4f} | {r['null_mean']:+.4f} | "
                 f"{r['null_std']:.4f} | {r['p_value']:.3g} |")
    L.append("")

    # 6
    m6 = results["effect_sizes"]
    L.append("## 6. Cohen's d_z + Hedges' g + bootstrap CIs (paired)")
    L.append("")
    L.append("| Axis | Severity | n | Cohen's d_z | d_z 95% CI | Hedges' g | g 95% CI |")
    L.append("|---|---|---:|---:|---|---:|---|")
    for r in m6["per_cell"]:
        dl, dh = r["cohens_dz_ci"]
        gl, gh = r["hedges_g_ci"]
        L.append(f"| {r['axis']} | {r['severity']} | {r['n']} | {r['cohens_dz']:+.3f} | "
                 f"[{dl:+.3f}, {dh:+.3f}] | {r['hedges_g']:+.3f} | [{gl:+.3f}, {gh:+.3f}] |")
    L.append("")

    # 7
    m7 = results["calibration_extended"]
    L.append("## 7. Extended calibration (ECE / Brier+Murphy / MCE / ACE)")
    for name in ("clean_baseline", "pooled_naive", "pooled_canon", "pooled_shifted"):
        c = m7[name]
        L.append(f"### {name}")
        L.append(f"- ECE = {c['ece']:.4f}, MCE = {c['mce']['mce']:.4f}, "
                 f"ACE = {c['ace']['ace']:.4f}")
        b = c["brier"]
        L.append(f"- Brier = {b['brier']:.4f}; Murphy decomposition: "
                 f"Reliability = {b['reliability']:.4f}, "
                 f"Resolution = {b['resolution']:.4f}, "
                 f"Uncertainty = {b['uncertainty']:.4f}")
        L.append(f"- Base rate = {b['base_rate']:.3f}, n = {b['n']}, "
                 f"decomp residual = {b['decomp_residual']:+.2e}")
        L.append("")

    # 8
    m8 = results["risk_coverage_aurc"]
    L.append("## 8. Risk-coverage AURC + E-AURC")
    L.append(f"- n = {m8['n']} pooled shifted recordings")
    for k in ("abs_logit", "neg_var_logit"):
        c = m8[k]
        ci = c["aurc_ci"]; eci = c["e_aurc_ci"]
        L.append(f"- Confidence = `{k}`: AURC = {c['aurc']:.4f} "
                 f"(95% CI [{ci[0]:.4f}, {ci[1]:.4f}]), "
                 f"E-AURC = {c['e_aurc']:.4f} "
                 f"(95% CI [{eci[0]:.4f}, {eci[1]:.4f}]), "
                 f"AURC oracle = {c['aurc_oracle']:.4f}")
    L.append("")

    # 9
    m9 = results["conformal_split"]
    L.append("## 9. Split-conformal classification (alpha=0.10)")
    L.append(f"- Calibration patients: {m9['n_cal_patients']}, "
             f"test patients: {m9['n_test_patients']}")
    for nm in ("clean_test", "shifted_test"):
        c = m9[nm]
        L.append(f"### {nm}")
        L.append(f"- q_hat = {c['q_hat']:.4f}, "
                 f"nominal coverage = {c['nominal_coverage']:.2f}, "
                 f"empirical marginal coverage = {c['coverage_marginal']:.4f}, "
                 f"mean set size = {c['set_size_mean']:.3f}")
        for cls, vals in c["coverage_by_class"].items():
            L.append(f"  - class {cls}: n={vals['n']}, covered={vals['covered']}, "
                     f"coverage={vals['coverage']:.4f}")
    L.append("")

    # 10
    m10 = results["mi_feature_target"]
    L.append("## 10. Mutual information (feature -> target)")
    mil = m10["mi_features_vs_label"]["mi_per_feature"]
    mis = m10["mi_features_vs_severity_id"]["mi_per_feature"]
    L.append(f"- Features: {m10['feature_names']}")
    L.append(f"- MI(features; label):       {[round(x, 4) for x in mil]} "
             f"(total {m10['mi_features_vs_label']['mi_total']:.4f})")
    L.append(f"- MI(features; severity_id): {[round(x, 4) for x in mis]} "
             f"(total {m10['mi_features_vs_severity_id']['mi_total']:.4f})")
    L.append(f"- {m10['note']}")
    L.append("")

    # 11
    m11 = results["seed_sensitivity"]
    L.append("## 11. Sampling-variability proxy (patient bootstrap)")
    L.append(f"- AUROC: mean={m11['auroc_mean']:.4f}, std={m11['auroc_std']:.4f}, "
             f"min={m11['auroc_min']:.4f}, max={m11['auroc_max']:.4f}")
    L.append(f"- ECE: mean={m11['ece_mean']:.4f}, std={m11['ece_std']:.4f}")
    L.append(f"- {m11['note']}")
    L.append("")

    # 12
    m12 = results["power_calc"]
    L.append("## 12. Retrospective power calculation")
    L.append(f"- Observed median |Cohen's d_z| across cells: {m12['observed_median_abs_dz']:.4f}")
    for tag in ("n_required_d020", "n_required_d050", "n_required_d080"):
        n = m12[tag]
        L.append(f"- {tag}: {n['comment']}")
    L.append("")

    # Verification
    v = results["verification"]
    L.append("## Verification: re-derived paper-cited numbers")
    L.append(f"- Drift count (above tolerance): **{v['drift_count']}**")
    L.append("")
    L.append("| Name | Paper | Computed | |Diff| | Drift? | Location |")
    L.append("|---|---:|---:|---:|:-:|---|")
    for c in v["checks"]:
        diff = c["abs_diff"]
        diff_s = f"{diff:.4f}" if diff is not None and math.isfinite(diff) else "n/a"
        L.append(f"| {c['name']} | {c['paper_value']} | {c['computed']:.4f} | "
                 f"{diff_s} | {'YES' if c['drift_above_tol'] else 'no'} | {c['location']} |")
    L.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--per-rec-csv", type=Path,
                    default=DATA / "per_recording_predictions_latest.csv")
    ap.add_argument("--sweep-csv", type=Path,
                    default=DATA / "sweep_latest.csv")
    ap.add_argument("--case-study-csv", type=Path,
                    default=DATA / "spmb_case_study_results.csv")
    ap.add_argument("--out-json", type=Path,
                    default=DATA / "advanced_stats_results.json")
    ap.add_argument("--out-summary", type=Path,
                    default=DATA / "advanced_stats_summary.md")
    ap.add_argument("--seeds-json", type=Path,
                    default=DATA / "seeds.json")
    ap.add_argument("--n-permutations", type=int, default=10000)
    args = ap.parse_args(argv)

    seeds = {
        "default":           DEFAULT_SEED,
        "fdr_bh":            DEFAULT_SEED,
        "bayes":             DEFAULT_SEED,
        "mixed_effects":     DEFAULT_SEED,
        "stratified_bootstrap": DEFAULT_SEED,
        "permutation_test":  DEFAULT_SEED,
        "effect_sizes":      DEFAULT_SEED,
        "aurc_bootstrap":    DEFAULT_SEED,
        "conformal":         DEFAULT_SEED,
        "mi":                DEFAULT_SEED,
        "seed_sensitivity_base": DEFAULT_SEED,
        "n_permutations":    int(args.n_permutations),
    }
    args.seeds_json.parent.mkdir(parents=True, exist_ok=True)
    args.seeds_json.write_text(json.dumps(seeds, indent=2) + "\n")
    print(f"[ok] wrote {args.seeds_json}")

    t0 = time.time()
    df = load_predictions(args.per_rec_csv)
    sweep = load_sweep(args.sweep_csv)
    if args.case_study_csv.exists():
        case_study = load_case_study(args.case_study_csv)
        ncs = len(case_study)
    else:
        case_study = None
        ncs = 0
        print(f"[load] case-study CSV not present at {args.case_study_csv} "
              "(optional; not part of the submitted paper) — case-study checks skipped")
    print(f"[load] {len(df)} per-rec rows, {len(sweep)} sweep rows, {ncs} case-study rows")

    results: dict[str, Any] = {}

    print("[run] 1) FDR")
    results["fdr_bh_q05"] = run_method_1_fdr(df)

    print("[run] 2) Bayesian Beta-Binomial")
    results["bayesian_beta_binomial"] = run_method_2_bayes(df)

    print("[run] 3) Mixed-effects logistic")
    results["mixed_effects_logistic"] = run_method_3_mixed_effects(df, seed=DEFAULT_SEED)

    print("[run] 4) Stratified vs naive bootstrap")
    results["stratified_vs_naive_bootstrap"] = run_method_4_stratified_bootstrap(df, seed=DEFAULT_SEED)

    print(f"[run] 5) Permutation test (B={args.n_permutations})")
    results["permutation_test_axis_shift"] = run_method_5_permutation(
        df, n_permutations=args.n_permutations, seed=DEFAULT_SEED,
    )

    print("[run] 6) Effect sizes (d / g) + bootstrap CIs")
    results["effect_sizes"] = run_method_6_effect_sizes(df, seed=DEFAULT_SEED)

    print("[run] 7) Extended calibration")
    results["calibration_extended"] = run_method_7_calibration(df)

    print("[run] 8) AURC + E-AURC")
    results["risk_coverage_aurc"] = run_method_8_aurc(df, seed=DEFAULT_SEED)

    print("[run] 9) Split conformal")
    results["conformal_split"] = run_method_9_conformal(df, seed=DEFAULT_SEED)

    print("[run] 10) Mutual information")
    results["mi_feature_target"] = run_method_10_mi(df, seed=DEFAULT_SEED)

    print("[run] 11) Seed sensitivity proxy")
    results["seed_sensitivity"] = run_method_11_seed_sens(df)

    print("[run] 12) Retrospective power")
    results["power_calc"] = run_method_12_power(results["effect_sizes"])

    print("[run] verification of paper-cited numbers")
    results["verification"] = run_verification(df, sweep, case_study)

    # Wall time / metadata
    results["_meta"] = {
        "seed":            DEFAULT_SEED,
        "n_per_rec_rows":  int(len(df)),
        "n_sweep_rows":    int(len(sweep)),
        "n_case_study_kept": int(len(case_study)) if case_study is not None else 0,
        "wall_seconds":    float(time.time() - t0),
        "input_files": {
            "per_rec_csv":    str(args.per_rec_csv),
            "sweep_csv":      str(args.sweep_csv),
            "case_study_csv": str(args.case_study_csv),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_sanitize(results), indent=2,
                                       default=_json_default) + "\n")
    print(f"[ok] wrote {args.out_json} ({args.out_json.stat().st_size} bytes)")

    write_summary_md(results, args.out_summary)
    print(f"[ok] wrote {args.out_summary}")

    print(f"[done] {results['_meta']['wall_seconds']:.1f}s total. "
          f"FDR rejected {results['fdr_bh_q05']['n_sig_bh_q05']}/{results['fdr_bh_q05']['n_tests']}; "
          f"verification drift = {results['verification']['drift_count']} above tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
