"""
Advanced statistical analyses for the SPMB 2026 acquisition-shift paper.

Every function in this module:
  * takes the per-recording / sweep CSV(s) (or numpy arrays) as input
  * returns a dict with the statistic + CI + diagnostics
  * uses a pinned seed (default 7) for any stochastic step
  * makes NO network calls and writes NO files (the orchestrator does I/O)

Tier 1 (must-add):
  1. fdr_benjamini_hochberg            -- BH FDR across axis x severity p-values
  2. bayesian_beta_binomial            -- Bernoulli-Beta posterior accuracy CIs per arm
  3. mixed_effects_logistic            -- patient-level random-intercept GLMM
                                          (sm.MixedLM Gaussian approximation +
                                           sm.BinomialBayesMixedGLM if available)
  4. stratified_bootstrap_vs_naive     -- class-stratified vs naive bootstrap on AUROC delta
  5. permutation_test_axis_shift       -- shuffle arm labels within recordings
  6. effect_sizes_d_g_with_ci          -- Cohen's d, Hedges' g, bootstrap CIs
  7. calibration_extended              -- Brier (Murphy decomp), MCE, ACE,
                                          binomial-CI reliability bins
  8. risk_coverage_aurc_eaurc          -- full AURC + excess-AURC (vs oracle) + CI

Tier 2 (best-effort):
  9. conformal_split_class_conditional -- class-conditional coverage at alpha=0.10
 10. mi_embedding_axis                 -- mutual info between log-prob features and
                                          (label, severity); embeddings not in CSV so
                                          we use the per-recording (logit, prob, var_logit)
                                          summary triplet as the smallest available
                                          embedding surrogate
 11. seed_sensitivity                  -- bootstrap-driven sensitivity proxy on
                                          patient-resampled splits (true 20-seed
                                          retrain not feasible from CSV alone)
 12. power_calc_retrospective          -- minimum N to detect d=0.2 at 80% power

DeLong + paired bootstrap on AUROC delta are NOT redefined here -- they live in
   apps/training/spmb_acquisition_shift/statistical_tests.py
The orchestrator (run_all_advanced_stats.py) re-uses the canonical implementation
if it can import that module; otherwise it falls back to a small local copy in
   _delong_compat.py
so this module never depends on any private repo being importable.

Reproducibility:
  Every function takes `seed` (default 7). The orchestrator emits seeds.json with
  the exact seeds used.

Author: Hitesh Dammu, 2026-06-25, for SPMB 2026 submission.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Standard library / scipy only at module top.
from scipy import stats


# =====================================================================
# Shared primitives
# =====================================================================
def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank (average rank for ties); equivalent to scipy.rankdata(method='average')."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Empirical AUROC via the Mann-Whitney U identity, with mid-rank ties."""
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


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    p = float((y_true == 1).sum())
    n = float((y_true == 0).sum())
    tpr = tp / p if p > 0 else 0.0
    tnr = tn / n if n > 0 else 0.0
    return 0.5 * (tpr + tnr)


# =====================================================================
# (1) FDR -- Benjamini-Hochberg correction
# =====================================================================
def fdr_benjamini_hochberg(
    pvals: np.ndarray,
    q: float = 0.05,
) -> dict:
    """Benjamini-Hochberg FDR procedure.

    Returns a dict with:
      adjusted_p:   (N,) ndarray of BH-adjusted p-values (monotone, in [0,1])
      reject:       (N,) bool, True iff adjusted_p[i] <= q
      threshold_p:  largest raw p-value still rejected (or 0.0 if none)
      n_tests:      N
      n_rejected:   number rejected

    Implementation matches Benjamini & Hochberg (1995) eq. (1):
      sort p in ascending order, let k* = max{k : p_(k) <= (k/N) * q},
      reject all H_(1)..H_(k*).
    Adjusted p_i = min over k>=rank(i) of (N/k) * p_(k), clamped to [0,1].
    """
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    if n == 0:
        return {"adjusted_p": np.array([]), "reject": np.array([], dtype=bool),
                "threshold_p": 0.0, "n_tests": 0, "n_rejected": 0, "q": q}
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    sorted_p = p[order]
    # raw BH scaled values, monotone non-increasing from the right
    bh_raw = sorted_p * n / ranks
    # enforce monotonicity: adjusted_p_(k) = min_{j>=k} bh_raw_(j)
    bh_mono = np.minimum.accumulate(bh_raw[::-1])[::-1]
    bh_clipped = np.clip(bh_mono, 0.0, 1.0)
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = bh_clipped
    reject = adjusted <= q
    if reject.any():
        threshold_p = float(p[reject].max())
    else:
        threshold_p = 0.0
    return {
        "adjusted_p": adjusted,
        "reject":     reject,
        "threshold_p": threshold_p,
        "n_tests":    int(n),
        "n_rejected": int(reject.sum()),
        "q":          float(q),
    }


# =====================================================================
# (2) Bayesian Bernoulli-Beta posterior on per-arm accuracy
# =====================================================================
def bayesian_beta_binomial(
    labels: np.ndarray,
    preds: np.ndarray,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    ci_level: float = 0.95,
) -> dict:
    """Bernoulli-Beta conjugate posterior on accuracy.

    Per-recording correctness c_i = 1{pred_i == label_i}. Under
    Bernoulli(theta) likelihood with a Beta(alpha0, beta0) prior, the
    posterior is Beta(alpha0 + k, beta0 + n - k) where k = sum(c) and
    n = len(c). Default prior is Beta(1,1) (Jeffreys-uniform / Bayes-Laplace).

    Returns posterior mean and equal-tailed 95% credible interval, plus
    the raw point-accuracy and Wilson 95% CI for direct comparison with
    the frequentist arm of the paper.
    """
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if labels.shape != preds.shape or labels.size == 0:
        return {"posterior_mean": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "accuracy": float("nan"),
                "wilson_lo": float("nan"), "wilson_hi": float("nan"),
                "n": 0, "k": 0,
                "prior_alpha": prior_alpha, "prior_beta": prior_beta,
                "ci_level": ci_level}
    correct = (labels == preds).astype(np.int64)
    n = int(correct.size)
    k = int(correct.sum())
    a = prior_alpha + k
    b = prior_beta + n - k
    alpha = 1.0 - ci_level
    lo = float(stats.beta.ppf(alpha / 2.0, a, b))
    hi = float(stats.beta.ppf(1.0 - alpha / 2.0, a, b))
    mean = a / (a + b)
    # Wilson CI on the raw fraction (no continuity correction)
    p_hat = k / n
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    wilson_lo = max(0.0, centre - half)
    wilson_hi = min(1.0, centre + half)
    return {
        "posterior_mean": float(mean),
        "ci_lo":          float(lo),
        "ci_hi":          float(hi),
        "accuracy":       float(p_hat),
        "wilson_lo":      float(wilson_lo),
        "wilson_hi":      float(wilson_hi),
        "n":              n,
        "k":              k,
        "prior_alpha":    float(prior_alpha),
        "prior_beta":     float(prior_beta),
        "ci_level":       float(ci_level),
    }


# =====================================================================
# (3) Mixed-effects logistic regression
# =====================================================================
def mixed_effects_logistic(
    df: pd.DataFrame,
    response: str = "correct",
    fixed_terms: Iterable[str] = ("arm", "severity_z"),
    group: str = "patient_id",
    method: str = "auto",
) -> dict:
    """Mixed-effects logistic regression with `group` as random intercept.

    Inputs:
      df: long-format DataFrame with columns [response, *fixed_terms, group]
          + an optional interaction column 'arm_x_severity_z' (if absent we
          create it from arm and severity_z).
      response: column name of the binary 0/1 outcome (per-row correctness).
      fixed_terms: iterable of fixed-effect column names.
      group: column name for random-intercept grouping (typically patient_id).
      method: 'binomialbayes' (statsmodels.BinomialBayesMixedGLM, true logit
              link, recommended), or 'mixedlm' (Gaussian-approx LMM on
              probabilities -- DEFINITELY a coarse fallback, flagged in the
              return value), or 'auto' (try BinomialBayesMixedGLM first,
              fall back to MixedLM).

    Returns a dict with:
      fixed_effects: dict of {term -> {'estimate','se','ci_lo','ci_hi','z','p'}}
      lrt_pvalue:    likelihood-ratio test vs intercept-only model (arm dropped)
      n_obs, n_groups, method_used
    """
    import statsmodels.api as sm  # local import keeps top-level fast

    df = df.copy()
    fixed_terms = list(fixed_terms)
    # Encode arm as numeric BEFORE assembling the design matrix.
    if "arm" in df.columns and "arm_canon" not in df.columns:
        df["arm_canon"] = (df["arm"] == "canonicalized").astype(np.float64)
    if "severity_z" in df.columns and "arm_canon" in df.columns and "arm_x_severity_z" not in df.columns:
        df["arm_x_severity_z"] = df["arm_canon"] * df["severity_z"].astype(float)
    # design matrix
    feature_cols: list[str] = []
    if "arm_canon" in df.columns:
        feature_cols.append("arm_canon")
    if "severity_z" in df.columns:
        feature_cols.append("severity_z")
    if "arm_x_severity_z" in df.columns:
        feature_cols.append("arm_x_severity_z")
    # drop any extra fixed terms the caller passed in that exist (excluding raw 'arm')
    for t in fixed_terms:
        if t == "arm":
            continue
        if t not in feature_cols and t in df.columns:
            feature_cols.append(t)
    feature_cols = list(dict.fromkeys(feature_cols))  # dedupe, preserve order
    X = sm.add_constant(df[feature_cols].astype(float).values, has_constant="add")
    y = df[response].astype(int).values
    groups = df[group].astype(str).values
    n_obs = int(len(df))
    n_groups = int(pd.Series(groups).nunique())

    method_used = None
    fe: dict = {}
    lrt_p = float("nan")

    # ----- Try BinomialBayesMixedGLM (true logit link, MAP fit) -----
    if method in ("auto", "binomialbayes"):
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

            # exog_vc / ident: simple random intercept per group
            exog_re = np.ones((n_obs, 1))
            ident = np.zeros(1, dtype=np.int64)
            model = BinomialBayesMixedGLM(
                endog=y,
                exog=X,
                exog_vc=exog_re,
                ident=ident,
                vc_names=[group],
            )
            res = model.fit_map()
            params = np.asarray(res.params, dtype=np.float64)
            # Posterior covariance from the Hessian (Laplace approx).
            try:
                cov = np.asarray(res.cov_params(), dtype=np.float64)
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except Exception:
                se = np.full_like(params, np.nan)
            term_names = ["const"] + feature_cols
            # BinomialBayesMixedGLM appends variance components at the end.
            for i, t in enumerate(term_names):
                if i >= len(params):
                    break
                est = float(params[i])
                s = float(se[i]) if i < len(se) else float("nan")
                if math.isfinite(s) and s > 0:
                    z = est / s
                    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
                    lo = est - 1.96 * s
                    hi = est + 1.96 * s
                else:
                    z, p, lo, hi = float("nan"), float("nan"), float("nan"), float("nan")
                fe[t] = {
                    "estimate": est, "se": s, "ci_lo": float(lo),
                    "ci_hi": float(hi), "z": float(z), "p": float(p),
                }
            method_used = "BinomialBayesMixedGLM (Laplace approx, MAP fit)"

            # Joint Wald test of all arm-related fixed effects against zero.
            # (BinomialBayesMixedGLM.fit_map() does not expose a robust
            # log-likelihood for an LRT; the joint Wald test is the
            # standard alternative for GLMM hypothesis testing.)
            arm_indices = [i for i, t in enumerate(["const"] + feature_cols)
                           if t in ("arm_canon", "arm_x_severity_z")]
            try:
                cov_full = np.asarray(res.cov_params(), dtype=np.float64)
                if arm_indices:
                    beta = params[arm_indices]
                    cov_sub = cov_full[np.ix_(arm_indices, arm_indices)]
                    cov_inv = np.linalg.pinv(cov_sub)
                    W = float(beta @ cov_inv @ beta)
                    dof = len(arm_indices)
                    lrt_p = float(1.0 - stats.chi2.cdf(W, df=dof))
                else:
                    lrt_p = float("nan")
            except Exception:
                lrt_p = float("nan")
            return {
                "fixed_effects": fe,
                "lrt_pvalue":    lrt_p,
                "n_obs":         n_obs,
                "n_groups":      n_groups,
                "method_used":   method_used,
            }
        except Exception as exc:  # noqa: BLE001
            if method == "binomialbayes":
                raise
            # Fall through to MixedLM
            method_used_fail = f"BinomialBayesMixedGLM failed: {exc!s}"

    # ----- Fallback: linear MixedLM on 0/1 outcome (linear probability model) -----
    try:
        mod = sm.MixedLM(endog=y.astype(float), exog=X, groups=groups)
        res = mod.fit(method="lbfgs", disp=False)
        term_names = ["const"] + feature_cols
        params = np.asarray(res.params)[:len(term_names)]
        se = np.asarray(res.bse)[:len(term_names)]
        for i, t in enumerate(term_names):
            est = float(params[i])
            s = float(se[i]) if i < len(se) else float("nan")
            if math.isfinite(s) and s > 0:
                z = est / s
                p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
                lo = est - 1.96 * s
                hi = est + 1.96 * s
            else:
                z, p, lo, hi = float("nan"), float("nan"), float("nan"), float("nan")
            fe[t] = {
                "estimate": est, "se": s, "ci_lo": float(lo),
                "ci_hi": float(hi), "z": float(z), "p": float(p),
            }
        # LRT vs null
        null_cols = [c for c in feature_cols if c not in ("arm_canon", "arm_x_severity_z")]
        if null_cols:
            X_null = sm.add_constant(df[null_cols].astype(float).values, has_constant="add")
        else:
            X_null = np.ones((n_obs, 1))
        null_res = sm.MixedLM(endog=y.astype(float), exog=X_null, groups=groups).fit(method="lbfgs", disp=False)
        ll_full = float(res.llf)
        ll_null = float(null_res.llf)
        dof = max(1, X.shape[1] - X_null.shape[1])
        lrt_stat = 2.0 * (ll_full - ll_null)
        lrt_p = float(1.0 - stats.chi2.cdf(lrt_stat, df=dof))
        method_used = "MixedLM (Gaussian linear-probability fallback)"
        return {
            "fixed_effects": fe,
            "lrt_pvalue":    lrt_p,
            "n_obs":         n_obs,
            "n_groups":      n_groups,
            "method_used":   method_used,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "fixed_effects": fe,
            "lrt_pvalue":    float("nan"),
            "n_obs":         n_obs,
            "n_groups":      n_groups,
            "method_used":   f"FAILED: {exc!s}",
        }


# =====================================================================
# (4) Stratified vs naive bootstrap on AUROC delta
# =====================================================================
def stratified_bootstrap_vs_naive(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 7,
) -> dict:
    """Paired bootstrap on Delta = AUROC(b) - AUROC(a), two variants.

    naive: resample row indices with replacement, ignoring class.
    stratified: resample with replacement separately within each class so the
                class prevalence in every resample matches the source set
                (canonical defensive choice for AUROC under class imbalance).

    Returns both percentile CIs and the absolute difference between their
    half-widths to quantify whether the choice changes inference.
    """
    labels = np.asarray(labels, dtype=np.int64)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    N = labels.size
    if N == 0:
        nan_out = (float("nan"), float("nan"))
        return {"naive_ci": nan_out, "stratified_ci": nan_out,
                "delta_point": float("nan"),
                "halfwidth_naive": float("nan"), "halfwidth_stratified": float("nan"),
                "halfwidth_diff": float("nan"),
                "n_resamples": int(n_resamples), "seed": int(seed)}
    delta_point = auroc(labels, b) - auroc(labels, a)
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    deltas_naive = np.empty(n_resamples, dtype=np.float64)
    deltas_strat = np.empty(n_resamples, dtype=np.float64)
    deltas_naive[:] = np.nan
    deltas_strat[:] = np.nan
    for i in range(n_resamples):
        # naive
        idx = rng.integers(0, N, size=N)
        yb = labels[idx]
        if len(np.unique(yb)) >= 2:
            deltas_naive[i] = auroc(yb, b[idx]) - auroc(yb, a[idx])
        # stratified by class
        idx_pos = pos_idx[rng.integers(0, n_pos, size=n_pos)] if n_pos > 0 else pos_idx
        idx_neg = neg_idx[rng.integers(0, n_neg, size=n_neg)] if n_neg > 0 else neg_idx
        idx_s = np.concatenate([idx_pos, idx_neg])
        yb_s = labels[idx_s]
        if len(np.unique(yb_s)) >= 2:
            deltas_strat[i] = auroc(yb_s, b[idx_s]) - auroc(yb_s, a[idx_s])
    deltas_naive = deltas_naive[~np.isnan(deltas_naive)]
    deltas_strat = deltas_strat[~np.isnan(deltas_strat)]
    alpha = 1.0 - ci_level
    def _ci(d):
        if d.size == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(d, 100.0 * alpha / 2.0)),
                float(np.percentile(d, 100.0 * (1.0 - alpha / 2.0))))
    naive_ci = _ci(deltas_naive)
    strat_ci = _ci(deltas_strat)
    hw_n = (naive_ci[1] - naive_ci[0]) / 2.0 if not any(np.isnan(naive_ci)) else float("nan")
    hw_s = (strat_ci[1] - strat_ci[0]) / 2.0 if not any(np.isnan(strat_ci)) else float("nan")
    hw_d = abs(hw_n - hw_s) if (math.isfinite(hw_n) and math.isfinite(hw_s)) else float("nan")
    return {
        "delta_point":          float(delta_point),
        "naive_ci":             naive_ci,
        "stratified_ci":        strat_ci,
        "halfwidth_naive":      float(hw_n),
        "halfwidth_stratified": float(hw_s),
        "halfwidth_diff":       float(hw_d),
        "n_resamples":          int(n_resamples),
        "seed":                 int(seed),
    }


# =====================================================================
# (5) Permutation test on axis-shift effect
# =====================================================================
def permutation_test_axis_shift(
    labels: np.ndarray,
    scores_naive: np.ndarray,
    scores_canon: np.ndarray,
    n_permutations: int = 10000,
    statistic: str = "delta_auroc",
    seed: int = 7,
) -> dict:
    """Within-recording permutation test on the arm effect.

    Null: scores_naive and scores_canon are exchangeable within each recording,
    i.e. the arm label has no effect on the score given the recording.

    For each permutation, for every recording independently flip a fair coin to
    swap (naive, canon) -> (canon, naive). Recompute the statistic.
    Empirical two-sided p-value: (1 + #{|T_perm| >= |T_obs|}) / (1 + B).

    Statistics supported:
      delta_auroc:   AUROC(scores_b) - AUROC(scores_a)  (b = canon by default)
    """
    labels = np.asarray(labels, dtype=np.int64)
    a = np.asarray(scores_naive, dtype=np.float64)
    b = np.asarray(scores_canon, dtype=np.float64)
    N = labels.size
    if N == 0:
        return {"observed": float("nan"), "p_value": float("nan"),
                "null_mean": float("nan"), "null_std": float("nan"),
                "n_permutations": int(n_permutations), "seed": int(seed)}
    if statistic != "delta_auroc":
        raise NotImplementedError(f"statistic={statistic!r} not supported")
    T_obs = auroc(labels, b) - auroc(labels, a)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    null[:] = np.nan
    for i in range(n_permutations):
        flips = rng.integers(0, 2, size=N).astype(bool)
        a_perm = np.where(flips, b, a)
        b_perm = np.where(flips, a, b)
        null[i] = auroc(labels, b_perm) - auroc(labels, a_perm)
    null = null[~np.isnan(null)]
    # two-sided
    p = float((1.0 + np.sum(np.abs(null) >= abs(T_obs))) / (1.0 + null.size))
    return {
        "observed":       float(T_obs),
        "p_value":        p,
        "null_mean":      float(null.mean()) if null.size else float("nan"),
        "null_std":       float(null.std(ddof=1)) if null.size > 1 else float("nan"),
        "n_permutations": int(null.size),
        "seed":           int(seed),
    }


# =====================================================================
# (6) Effect sizes: Cohen's d, Hedges' g, with bootstrap CIs
# =====================================================================
def cohens_d_paired(x_a: np.ndarray, x_b: np.ndarray) -> float:
    """Paired Cohen's d_z = mean(b-a) / sd(b-a), as in Lakens (2013)."""
    a = np.asarray(x_a, dtype=np.float64)
    b = np.asarray(x_b, dtype=np.float64)
    d = b - a
    if d.size < 2:
        return 0.0
    sd = float(d.std(ddof=1))
    return 0.0 if sd == 0.0 else float(d.mean() / sd)


def hedges_g_from_d(d: float, n: int) -> float:
    """Hedges' g = d * J(n), with J(n) = 1 - 3 / (4*(n-1) - 1).

    For paired d_z the degrees of freedom term is n_pairs - 1; here we pass
    n_pairs as n.
    """
    if n <= 1:
        return float(d)
    J = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    return float(d * J)


def effect_sizes_d_g_with_ci(
    x_a: np.ndarray,
    x_b: np.ndarray,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 7,
) -> dict:
    """Paired Cohen's d_z + Hedges' g + bootstrap percentile CIs on both.

    Bootstrap resamples paired rows (so the pairing is preserved).
    """
    a = np.asarray(x_a, dtype=np.float64)
    b = np.asarray(x_b, dtype=np.float64)
    N = a.size
    d_point = cohens_d_paired(a, b)
    g_point = hedges_g_from_d(d_point, N)
    rng = np.random.default_rng(seed)
    d_boot = np.empty(n_resamples, dtype=np.float64)
    g_boot = np.empty(n_resamples, dtype=np.float64)
    d_boot[:] = np.nan
    g_boot[:] = np.nan
    for i in range(n_resamples):
        idx = rng.integers(0, N, size=N)
        d_i = cohens_d_paired(a[idx], b[idx])
        g_i = hedges_g_from_d(d_i, N)
        d_boot[i] = d_i
        g_boot[i] = g_i
    d_boot = d_boot[~np.isnan(d_boot)]
    g_boot = g_boot[~np.isnan(g_boot)]
    alpha = 1.0 - ci_level
    def _ci(arr):
        if arr.size == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(arr, 100.0 * alpha / 2.0)),
                float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0))))
    return {
        "n_pairs":         int(N),
        "cohens_dz":       float(d_point),
        "cohens_dz_ci":    _ci(d_boot),
        "hedges_g":        float(g_point),
        "hedges_g_ci":     _ci(g_boot),
        "small_sample_correction_factor_J": float(1.0 - 3.0 / (4.0 * max(N, 2) - 5.0)) if N > 1 else 1.0,
        "n_resamples":     int(n_resamples),
        "seed":            int(seed),
    }


# =====================================================================
# (7) Calibration: Brier (Murphy decomp), MCE, ACE, reliability CI bins
# =====================================================================
def brier_with_murphy_decomposition(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Brier score = Reliability - Resolution + Uncertainty (Murphy 1973).

    Reliability (lower=better): mean weighted squared distance between bin's
                                mean prob and bin's empirical accuracy.
    Resolution  (higher=better): variance of bin accuracies around the base rate.
    Uncertainty (irreducible):  base_rate * (1 - base_rate).

    Returns the four components + per-bin diagnostics. Uses equal-width binning.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    N = labels.size
    if N == 0:
        return {"brier": float("nan"),
                "reliability": float("nan"),
                "resolution":  float("nan"),
                "uncertainty": float("nan"),
                "decomp_residual": float("nan"),
                "n_bins": int(n_bins), "n": 0}
    base = float(labels.mean())
    brier = float(np.mean((probs - labels) ** 2))
    # Equal-width bins
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    rel = 0.0
    res = 0.0
    bins = []
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            bins.append({"bin": k, "n": 0, "mean_prob": float("nan"),
                         "accuracy": float("nan")})
            continue
        f_k = float(probs[mask].mean())
        o_k = float(labels[mask].mean())
        rel += nk / N * (f_k - o_k) ** 2
        res += nk / N * (o_k - base) ** 2
        bins.append({"bin": k, "n": nk, "mean_prob": f_k, "accuracy": o_k})
    unc = base * (1.0 - base)
    decomp_residual = brier - (rel - res + unc)
    return {
        "brier":           float(brier),
        "reliability":     float(rel),
        "resolution":      float(res),
        "uncertainty":     float(unc),
        "decomp_residual": float(decomp_residual),
        "base_rate":       float(base),
        "n_bins":          int(n_bins),
        "n":               int(N),
        "bins":            bins,
    }


def max_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    N = labels.size
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    gaps = []
    for k in range(n_bins):
        mask = bin_idx == k
        if not mask.any():
            continue
        gaps.append(abs(float(probs[mask].mean()) - float(labels[mask].mean())))
    return {"mce": float(max(gaps)) if gaps else float("nan"),
            "n_bins": int(n_bins), "n": int(N)}


def adaptive_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict:
    """Adaptive Calibration Error -- equal-mass (quantile) bins.

    Nguyen & O'Connor (2015) / Nixon et al. (2019).
    Reduces the heavy weight on the dense [0, 0.05] region that ECE puts
    on confident-correct predictions, by ensuring every bin has ~N/n_bins
    samples.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    N = labels.size
    if N == 0:
        return {"ace": float("nan"), "n_bins": int(n_bins), "n": 0}
    # quantile edges
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(probs, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # internal edges only for digitize
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    ace = 0.0
    used = 0
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        gap = abs(float(probs[mask].mean()) - float(labels[mask].mean()))
        ace += nk / N * gap
        used += 1
    return {"ace": float(ace), "n_bins": int(n_bins), "n_bins_used": used, "n": int(N)}


def reliability_diagram_with_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
    ci_level: float = 0.95,
) -> dict:
    """Per-bin (mean_prob, empirical_acc, [Wilson CI lo, hi], n).

    Use 95% Wilson interval per bin so reviewers see the binomial uncertainty
    on each reliability point rather than just the point.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    z = stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0)
    bins = []
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            bins.append({"bin": k, "n": 0,
                         "mean_prob": float("nan"),
                         "accuracy": float("nan"),
                         "ci_lo": float("nan"), "ci_hi": float("nan"),
                         "bin_edge_lo": float(edges[k]),
                         "bin_edge_hi": float(edges[k+1])})
            continue
        f_k = float(probs[mask].mean())
        o_k = float(labels[mask].mean())
        denom = 1.0 + z * z / nk
        centre = (o_k + z * z / (2.0 * nk)) / denom
        half = z * math.sqrt(o_k * (1.0 - o_k) / nk + z * z / (4.0 * nk * nk)) / denom
        bins.append({"bin": k, "n": nk,
                     "mean_prob": f_k, "accuracy": o_k,
                     "ci_lo": float(max(0.0, centre - half)),
                     "ci_hi": float(min(1.0, centre + half)),
                     "bin_edge_lo": float(edges[k]),
                     "bin_edge_hi": float(edges[k+1])})
    return {"bins": bins, "n_bins": int(n_bins), "ci_level": float(ci_level)}


def calibration_extended(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
    ci_level: float = 0.95,
) -> dict:
    """All calibration metrics in one call: ECE, Brier+Murphy, MCE, ACE, bins."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    # ECE (10-bin equal-width) for parity with paper
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    N = labels.size
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        ece += nk / N * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return {
        "ece":  float(ece),
        "brier": brier_with_murphy_decomposition(labels, probs, n_bins=n_bins),
        "mce":   max_calibration_error(labels, probs, n_bins=n_bins),
        "ace":   adaptive_calibration_error(labels, probs, n_bins=n_bins),
        "reliability_bins": reliability_diagram_with_ci(labels, probs, n_bins=n_bins, ci_level=ci_level),
        "n_bins": int(n_bins),
        "n":      int(N),
    }


# =====================================================================
# (8) Risk-coverage curve + AURC + E-AURC
# =====================================================================
def risk_coverage_aurc_eaurc(
    labels: np.ndarray,
    preds: np.ndarray,
    confidence: np.ndarray,
    seed: int = 7,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """Selective-prediction risk-coverage curve, AURC, and excess-AURC.

    confidence:  per-recording score where HIGHER means MORE trustworthy.
    preds:       binary predictions in {0,1}.
    labels:      binary ground truth in {0,1}.

    Procedure:
      Sort by descending confidence. For each coverage k/N, risk_k =
      (1/k) * sum_{i in top-k} 1{pred_i != label_i}.
      AURC = (1/N) * sum_k risk_k    (trapezoidal-equivalent on uniform x).
      E-AURC = AURC - AURC_oracle, where AURC_oracle ranks samples by the
              indicator 1{pred == label} (correct first, then errors).

    Bootstrap (paired-by-recording) gives the CI on AURC and E-AURC.
    """
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    conf = np.asarray(confidence, dtype=np.float64)
    N = labels.size
    if N == 0:
        return {"aurc": float("nan"), "e_aurc": float("nan"),
                "aurc_oracle": float("nan"),
                "aurc_ci": (float("nan"), float("nan")),
                "e_aurc_ci": (float("nan"), float("nan")),
                "n": 0, "n_bootstrap": int(n_bootstrap), "seed": int(seed),
                "curve": []}
    err = (preds != labels).astype(np.float64)

    def _aurc(conf_, err_):
        # sort descending by conf -> top-k corresponds to highest confidence kept.
        order = np.argsort(-conf_, kind="mergesort")
        err_sorted = err_[order]
        cum_err = np.cumsum(err_sorted)
        ks = np.arange(1, N + 1, dtype=np.float64)
        risks = cum_err / ks
        return float(risks.mean()), risks, order

    aurc_point, risks_curve, order = _aurc(conf, err)
    # Oracle: rank correct samples (err=0) before errors (err=1).
    aurc_oracle, _, _ = _aurc(-err, err)  # -err -> correct first
    e_aurc_point = aurc_point - aurc_oracle

    # paired bootstrap
    rng = np.random.default_rng(seed)
    aurc_boot = np.empty(n_bootstrap, dtype=np.float64)
    e_aurc_boot = np.empty(n_bootstrap, dtype=np.float64)
    aurc_boot[:] = np.nan
    e_aurc_boot[:] = np.nan
    for i in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        a_i, _, _ = _aurc(conf[idx], err[idx])
        o_i, _, _ = _aurc(-err[idx], err[idx])
        aurc_boot[i] = a_i
        e_aurc_boot[i] = a_i - o_i
    aurc_boot = aurc_boot[~np.isnan(aurc_boot)]
    e_aurc_boot = e_aurc_boot[~np.isnan(e_aurc_boot)]
    alpha = 1.0 - ci_level
    def _ci(arr):
        if arr.size == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(arr, 100.0 * alpha / 2.0)),
                float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0))))
    aurc_ci = _ci(aurc_boot)
    e_aurc_ci = _ci(e_aurc_boot)
    # downsample curve to <= 200 points for JSON portability
    if N > 200:
        idx_keep = np.linspace(0, N - 1, 200, dtype=int)
    else:
        idx_keep = np.arange(N)
    curve = [{"coverage": float((k + 1) / N), "risk": float(risks_curve[k])}
             for k in idx_keep]
    return {
        "aurc":         float(aurc_point),
        "aurc_oracle":  float(aurc_oracle),
        "e_aurc":       float(e_aurc_point),
        "aurc_ci":      aurc_ci,
        "e_aurc_ci":    e_aurc_ci,
        "n":            int(N),
        "n_bootstrap":  int(n_bootstrap),
        "seed":         int(seed),
        "curve":        curve,
    }


# =====================================================================
# (9) Tier 2: split conformal classification with class-conditional coverage
# =====================================================================
def conformal_split_class_conditional(
    cal_labels: np.ndarray,
    cal_probs: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """Split-conformal classification with the APS / softmax-residual score.

    Score s_i = 1 - prob_assigned_to_true_class_i (lower = better calibrated).
    Threshold q_hat = ceil((n_cal+1)(1-alpha)) / n_cal -quantile of cal scores.
    Test set: emit predicted SET {c : 1 - prob_c <= q_hat}.

    Empirical coverage = fraction of test points whose true label is in the set.
    Class-conditional coverage: empirical coverage within each class slice.
    """
    cal_labels = np.asarray(cal_labels, dtype=np.int64)
    cal_probs = np.asarray(cal_probs, dtype=np.float64)  # P(class=1)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    test_probs = np.asarray(test_probs, dtype=np.float64)
    if cal_labels.size == 0 or test_labels.size == 0:
        return {"q_hat": float("nan"), "coverage_marginal": float("nan"),
                "coverage_by_class": {}, "alpha": float(alpha),
                "n_cal": 0, "n_test": 0}
    # Per-row scores on calibration set: 1 - prob_true_class
    cal_prob_true = np.where(cal_labels == 1, cal_probs, 1.0 - cal_probs)
    cal_scores = 1.0 - cal_prob_true
    n_cal = len(cal_scores)
    # finite-sample quantile (Romano et al. 2019 correction)
    rank = math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    rank = min(1.0, max(0.0, rank))
    q_hat = float(np.quantile(cal_scores, rank, method="higher"))
    # On test: include class c iff 1 - P(c) <= q_hat, i.e. P(c) >= 1 - q_hat
    thr = 1.0 - q_hat
    set_size_avg = 0.0
    covered = 0
    cov_by_class = {0: {"covered": 0, "n": 0}, 1: {"covered": 0, "n": 0}}
    test_set_includes_true = np.zeros(len(test_labels), dtype=bool)
    set_size = np.zeros(len(test_labels), dtype=np.int64)
    for i, (y, p) in enumerate(zip(test_labels, test_probs)):
        included = []
        if p >= thr:
            included.append(1)
        if (1.0 - p) >= thr:
            included.append(0)
        set_size[i] = len(included)
        if int(y) in included:
            test_set_includes_true[i] = True
            covered += 1
        cov_by_class[int(y)]["n"] += 1
        if int(y) in included:
            cov_by_class[int(y)]["covered"] += 1
    set_size_avg = float(set_size.mean())
    cov_marg = float(covered / len(test_labels))
    cov_by_class_pct = {
        str(c): {
            "n":        v["n"],
            "covered":  v["covered"],
            "coverage": (v["covered"] / v["n"]) if v["n"] > 0 else float("nan"),
        }
        for c, v in cov_by_class.items()
    }
    return {
        "alpha":               float(alpha),
        "q_hat":               float(q_hat),
        "nominal_coverage":    float(1.0 - alpha),
        "coverage_marginal":   cov_marg,
        "coverage_by_class":   cov_by_class_pct,
        "set_size_mean":       set_size_avg,
        "set_size_distribution": dict(zip(*np.unique(set_size, return_counts=True))),
        "n_cal":               int(n_cal),
        "n_test":              int(len(test_labels)),
    }


# =====================================================================
# (10) Tier 2: Mutual information between (logit, prob, var_logit) and label/severity
# =====================================================================
def mi_embedding_axis(
    features: np.ndarray,
    target: np.ndarray,
    seed: int = 7,
) -> dict:
    """sklearn.feature_selection.mutual_info_classif on a feature matrix.

    The paper does not export EEGPT embeddings to the CSV; the smallest
    available per-recording representation is the (logit, prob, var_logit)
    triplet emitted by eval_loop.py. We use this surrogate, note the
    limitation in the report, and return per-feature MI plus the total.
    """
    from sklearn.feature_selection import mutual_info_classif
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[0] == 0:
        return {"mi_per_feature": [], "mi_total": float("nan"),
                "n": 0, "seed": int(seed)}
    mi = mutual_info_classif(features, target,
                             random_state=seed, n_neighbors=3)
    return {
        "mi_per_feature": [float(m) for m in mi],
        "mi_total":       float(np.sum(mi)),
        "n":              int(features.shape[0]),
        "seed":           int(seed),
    }


# =====================================================================
# (11) Tier 2: seed-sensitivity proxy on patient-resampled splits
# =====================================================================
def seed_sensitivity(
    labels: np.ndarray,
    probs: np.ndarray,
    n_seeds: int = 20,
    base_seed: int = 7,
) -> dict:
    """Patient-bootstrap proxy for seed sensitivity.

    A true re-training across 20 seeds is not possible from a CSV. The closest
    legitimate proxy we can run is: for each seed in [base_seed, base_seed+1,
    ..., base_seed+n_seeds-1], draw a paired bootstrap of recordings and
    recompute AUROC + ECE. The spread of AUROC across these bootstraps is a
    lower bound on the sampling variability of the original split; if the
    spread is small, the seed=7 point estimate is not a lucky pick under
    patient resampling. We explicitly flag this as a sampling-variability
    proxy in the report.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    N = labels.size
    aurocs = []
    eces = []
    for k in range(n_seeds):
        rng = np.random.default_rng(base_seed + k)
        idx = rng.integers(0, N, size=N)
        yb = labels[idx]
        pb = probs[idx]
        if len(np.unique(yb)) < 2:
            continue
        aurocs.append(auroc(yb, pb))
        eces.append(calibration_extended(yb, pb)["ece"])
    aurocs = np.asarray(aurocs)
    eces = np.asarray(eces)
    return {
        "n_seeds":         int(n_seeds),
        "base_seed":       int(base_seed),
        "auroc_mean":      float(aurocs.mean()) if aurocs.size else float("nan"),
        "auroc_std":       float(aurocs.std(ddof=1)) if aurocs.size > 1 else float("nan"),
        "auroc_min":       float(aurocs.min()) if aurocs.size else float("nan"),
        "auroc_max":       float(aurocs.max()) if aurocs.size else float("nan"),
        "ece_mean":        float(eces.mean()) if eces.size else float("nan"),
        "ece_std":         float(eces.std(ddof=1)) if eces.size > 1 else float("nan"),
        "note":            "patient-bootstrap proxy for sampling variability; "
                           "true 20-seed retrain requires re-running training,"
                           " not feasible from the CSV layer alone",
    }


# =====================================================================
# (12) Tier 2: retrospective power calculation
# =====================================================================
def power_calc_retrospective(
    observed_d: float,
    target_d: float = 0.2,
    power: float = 0.80,
    alpha: float = 0.05,
) -> dict:
    """Minimum N (paired) to detect a Cohen's d at the given power.

    For a paired t-test we use the standard formula with normal approximation:
      n >= ((z_{1-alpha/2} + z_{1-beta}) / d)^2
    A small-n adjustment is applied: t-distribution critical values via
    iterative search using scipy.stats.nct.
    """
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    z_b = stats.norm.ppf(power)
    if target_d <= 0:
        return {"n_required_normal": float("inf"), "n_required_t": float("inf"),
                "target_d": float(target_d), "observed_d": float(observed_d),
                "power": float(power), "alpha": float(alpha)}
    n_normal = math.ceil(((z_a + z_b) / target_d) ** 2)
    # Refine using noncentral-t: smallest n with NCT(df=n-1, nc=d*sqrt(n))
    # cdf at t_{1-alpha/2, n-1} achieving power.
    n = max(3, n_normal)
    while n < 100000:
        df = n - 1
        nc = target_d * math.sqrt(n)
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df)
        # two-sided power approx by one tail
        achieved = 1.0 - stats.nct.cdf(t_crit, df=df, nc=nc) + stats.nct.cdf(-t_crit, df=df, nc=nc)
        if achieved >= power:
            break
        n += 1
    return {
        "target_d":         float(target_d),
        "observed_d":       float(observed_d),
        "power":            float(power),
        "alpha":            float(alpha),
        "n_required_normal": int(n_normal),
        "n_required_t":     int(n),
        "comment": (
            f"To detect d={target_d} at {int(power*100)}% power (two-sided alpha={alpha}), "
            f"need >= {n} paired recordings (noncentral-t exact; "
            f"normal approx gives {n_normal})."
        ),
    }


# =====================================================================
# Utility: per-axis DeLong p-values for FDR pooling
# =====================================================================
def _delong_p(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Two-sided DeLong p, pure numpy (Sun & Xu 2014)."""
    labels = np.asarray(labels, dtype=np.int64)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    pos = labels == 1
    m = int(pos.sum())
    n = int((~pos).sum())
    if m == 0 or n == 0:
        return float("nan")
    arms = np.vstack([a, b])
    theta = np.empty(2)
    V10 = np.empty((2, m))
    V01 = np.empty((2, n))
    for k in range(2):
        x = arms[k, pos]
        y = arms[k, ~pos]
        tx = _midrank(x)
        ty = _midrank(y)
        tz = _midrank(np.concatenate([x, y]))
        V10[k] = (tz[:m] - tx) / float(n)
        V01[k] = 1.0 - (tz[m:] - ty) / float(m)
        theta[k] = V10[k].mean()
    S10 = np.cov(V10, ddof=1) if m > 1 else np.zeros((2, 2))
    S01 = np.cov(V01, ddof=1) if n > 1 else np.zeros((2, 2))
    S = S10 / float(m) + S01 / float(n)
    c = np.array([1.0, -1.0])
    var = float(c @ S @ c)
    diff = float(theta[0] - theta[1])
    if not np.isfinite(var) or var <= 0.0:
        return 1.0
    z = diff / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


__all__ = [
    "auroc",
    "fdr_benjamini_hochberg",
    "bayesian_beta_binomial",
    "mixed_effects_logistic",
    "stratified_bootstrap_vs_naive",
    "permutation_test_axis_shift",
    "effect_sizes_d_g_with_ci",
    "calibration_extended",
    "brier_with_murphy_decomposition",
    "max_calibration_error",
    "adaptive_calibration_error",
    "reliability_diagram_with_ci",
    "risk_coverage_aurc_eaurc",
    "conformal_split_class_conditional",
    "mi_embedding_axis",
    "seed_sensitivity",
    "power_calc_retrospective",
    "cohens_d_paired",
    "hedges_g_from_d",
    "_delong_p",
]
