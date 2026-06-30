"""
Unit tests for the SPMB 2026 statistical-rigor block (paper §Results,
statistical-tests addendum).

Covers the four primitives in
``apps.training.spmb_acquisition_shift.statistical_tests``:

  * auroc_paired        — compared against sklearn.metrics.roc_auc_score
                          on the same inputs (when sklearn is importable)
  * delong_p_value      — synthetic edge cases with known answers
  * paired_bootstrap_auroc_delta_ci — coverage + monotonicity properties
  * paired_cohens_d     — sign, scale, and zero-variance fallback

Plus an end-to-end CSV roundtrip:
  compute_statistical_tests() consumes a hand-built per-recording CSV
  and emits all expected columns with the right shape.

No real EDFs, no real model — pure numpy.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the src/ package directory is importable.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from statistical_tests import (  # noqa: E402
    auroc_paired,
    compute_statistical_tests,
    delong_p_value,
    load_per_recording_csv,
    paired_bootstrap_auroc_delta_ci,
    paired_cohens_d,
    pair_rows_by_recording,
    significance_marker,
    write_statistical_tests_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _balanced_labels(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int64)
    y[: n // 2] = 1
    rng.shuffle(y)
    return y


def _scores_for_labels(labels: np.ndarray, separation: float, seed: int) -> np.ndarray:
    """Gaussian scores: positives ~ N(separation, 1), negatives ~ N(0, 1)."""
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 1.0, size=labels.shape[0])
    s[labels == 1] += separation
    return s


# ---------------------------------------------------------------------------
# auroc_paired
# ---------------------------------------------------------------------------
class TestAurocPaired:
    def test_perfect_predictions_auroc_is_one(self):
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        scores = labels.astype(np.float64)   # predictions == labels
        assert auroc_paired(labels, scores) == pytest.approx(1.0)

    def test_perfectly_wrong_predictions_auroc_is_zero(self):
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        scores = (1 - labels).astype(np.float64)
        assert auroc_paired(labels, scores) == pytest.approx(0.0)

    def test_random_predictions_auroc_near_half(self):
        labels = _balanced_labels(n=2000, seed=1)
        rng = np.random.default_rng(2)
        scores = rng.normal(0.0, 1.0, size=labels.shape[0])
        # With n=2000 the empirical AUROC should be within ~0.03 of 0.5.
        assert abs(auroc_paired(labels, scores) - 0.5) < 0.03

    def test_single_class_returns_nan(self):
        labels = np.zeros(10, dtype=np.int64)
        scores = np.arange(10, dtype=np.float64)
        assert np.isnan(auroc_paired(labels, scores))

    def test_matches_sklearn_when_available(self):
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            pytest.skip("sklearn not installed in this env")
        labels = _balanced_labels(n=400, seed=42)
        scores = _scores_for_labels(labels, separation=0.8, seed=43)
        ours = auroc_paired(labels, scores)
        ref = roc_auc_score(labels, scores)
        assert ours == pytest.approx(ref, abs=1e-9)

    def test_ties_are_handled_with_midranks(self):
        # Half the positives tie with half the negatives → AUROC = 0.5
        # only if mid-rank averaging is used.
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        scores = np.array([0.5, 1.0, 0.5, 1.0], dtype=np.float64)
        # Pos ranks (mid-rank): both 1.5 + 3.5 = ... let's compute manually:
        #   sorted scores: [0.5, 0.5, 1.0, 1.0]
        #   mid-ranks:    [1.5, 1.5, 3.5, 3.5]
        # Pos at score=0.5 → rank 1.5; pos at score=1.0 → rank 3.5.
        # AUROC = (1.5 + 3.5 - 1*2/2 - 1*2/2 + ... )/(m*n)
        # Standard sklearn answer is 0.5 because the two arms are symmetric.
        assert auroc_paired(labels, scores) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# delong_p_value
# ---------------------------------------------------------------------------
class TestDeLongPValue:
    def test_identical_scores_gives_p_one(self):
        labels = _balanced_labels(n=200, seed=11)
        scores = _scores_for_labels(labels, separation=1.0, seed=12)
        auc_a, auc_b, p, z = delong_p_value(labels, scores, scores)
        assert auc_a == pytest.approx(auc_b)
        # Identical inputs → variance of difference is 0 → p clamped to 1
        assert p == pytest.approx(1.0)
        assert z == pytest.approx(0.0)

    def test_clearly_different_arms_significant(self):
        labels = _balanced_labels(n=500, seed=21)
        # Arm A: near-random; Arm B: very strong signal.
        scores_a = _scores_for_labels(labels, separation=0.1, seed=22)
        scores_b = _scores_for_labels(labels, separation=2.5, seed=23)
        auc_a, auc_b, p, z = delong_p_value(labels, scores_a, scores_b)
        assert auc_b > auc_a + 0.2     # arm B should win by a wide margin
        assert p < 1e-6
        assert z < 0                   # since theta_a - theta_b < 0

    def test_subtle_difference_with_large_n_is_significant(self):
        labels = _balanced_labels(n=800, seed=31)
        scores_a = _scores_for_labels(labels, separation=0.6, seed=32)
        scores_b = _scores_for_labels(labels, separation=0.9, seed=33)
        auc_a, auc_b, p, _z = delong_p_value(labels, scores_a, scores_b)
        # Arm B should still be better in expectation.
        assert auc_b > auc_a
        # And on 800 paired samples we should reach p < 0.05.
        assert p < 0.05

    def test_p_value_symmetric_in_argument_order(self):
        labels = _balanced_labels(n=300, seed=41)
        a = _scores_for_labels(labels, separation=0.8, seed=42)
        b = _scores_for_labels(labels, separation=1.3, seed=43)
        _, _, p_ab, z_ab = delong_p_value(labels, a, b)
        _, _, p_ba, z_ba = delong_p_value(labels, b, a)
        assert p_ab == pytest.approx(p_ba)
        assert z_ab == pytest.approx(-z_ba)

    def test_p_value_in_unit_interval(self):
        labels = _balanced_labels(n=80, seed=51)
        a = _scores_for_labels(labels, separation=0.4, seed=52)
        b = _scores_for_labels(labels, separation=0.6, seed=53)
        _, _, p, _z = delong_p_value(labels, a, b)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# paired_bootstrap_auroc_delta_ci
# ---------------------------------------------------------------------------
class TestPairedBootstrap:
    def test_identical_scores_ci_contains_zero(self):
        labels = _balanced_labels(n=120, seed=61)
        scores = _scores_for_labels(labels, separation=1.0, seed=62)
        lo, hi = paired_bootstrap_auroc_delta_ci(
            labels, scores, scores, n_resamples=500, seed=63
        )
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert hi == pytest.approx(0.0, abs=1e-9)

    def test_clearly_better_arm_ci_excludes_zero(self):
        labels = _balanced_labels(n=300, seed=71)
        a = _scores_for_labels(labels, separation=0.2, seed=72)
        b = _scores_for_labels(labels, separation=2.0, seed=73)
        lo, hi = paired_bootstrap_auroc_delta_ci(
            labels, a, b, n_resamples=500, seed=74
        )
        assert lo > 0.0       # CI on Δ = AUROC_b - AUROC_a strictly positive

    def test_ci_lo_le_hi_and_finite(self):
        labels = _balanced_labels(n=150, seed=81)
        a = _scores_for_labels(labels, separation=0.5, seed=82)
        b = _scores_for_labels(labels, separation=1.0, seed=83)
        lo, hi = paired_bootstrap_auroc_delta_ci(
            labels, a, b, n_resamples=200, seed=84
        )
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi

    def test_empty_input_returns_nan_pair(self):
        lo, hi = paired_bootstrap_auroc_delta_ci(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            n_resamples=10,
        )
        assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# paired_cohens_d
# ---------------------------------------------------------------------------
class TestCohensD:
    def test_zero_difference_returns_zero(self):
        a = np.array([1, 0, 1, 0, 1], dtype=np.float64)
        b = a.copy()
        assert paired_cohens_d(a, b) == 0.0

    def test_positive_when_b_better_on_average(self):
        # On 10 recordings: b is correct on 7, a on 3. Difference vector
        # has 4 positive ones and 0 negative ones (the other 6 agree).
        a = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 0], dtype=np.float64)
        b = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.float64)
        d = paired_cohens_d(a, b)
        assert d > 0

    def test_negative_when_b_worse_on_average(self):
        a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.float64)
        b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=np.float64)
        d = paired_cohens_d(a, b)
        assert d < 0

    def test_shape_mismatch_returns_nan(self):
        d = paired_cohens_d(np.array([0.0, 1.0]), np.array([0.0]))
        assert np.isnan(d)

    def test_empty_returns_nan(self):
        d = paired_cohens_d(np.array([]), np.array([]))
        assert np.isnan(d)


# ---------------------------------------------------------------------------
# significance_marker
# ---------------------------------------------------------------------------
class TestSignificanceMarker:
    @pytest.mark.parametrize("p, expected", [
        (0.0009, "***"),
        (0.005,  "**"),
        (0.04,   "*"),
        (0.5,    "n.s."),
        (1.0,    "n.s."),
        (float("nan"), "n/a"),
    ])
    def test_threshold_boundaries(self, p, expected):
        assert significance_marker(p) == expected


# ---------------------------------------------------------------------------
# pair_rows_by_recording
# ---------------------------------------------------------------------------
class TestPairing:
    def test_only_shared_rec_ids_kept(self):
        rows = [
            {"rec_id": "r1", "axis": "ax", "severity": "s", "arm": "naive",
             "label": 1, "logit": 0.1, "prob": 0.6},
            {"rec_id": "r1", "axis": "ax", "severity": "s", "arm": "canonicalized",
             "label": 1, "logit": 0.4, "prob": 0.8},
            {"rec_id": "r2", "axis": "ax", "severity": "s", "arm": "naive",
             "label": 0, "logit": -0.5, "prob": 0.3},
            # r2 has no canonicalized partner — should be dropped
            {"rec_id": "r3", "axis": "ax", "severity": "s", "arm": "canonicalized",
             "label": 0, "logit": -0.2, "prob": 0.45},
        ]
        labels, p_n, p_c, ids = pair_rows_by_recording(rows, "ax", "s")
        assert ids == ["r1"]
        assert labels.tolist() == [1]
        assert p_n.tolist() == pytest.approx([0.6])
        assert p_c.tolist() == pytest.approx([0.8])


# ---------------------------------------------------------------------------
# End-to-end CSV roundtrip
# ---------------------------------------------------------------------------
class TestComputeStatisticalTestsEndToEnd:
    def _build_synthetic_csv(self, tmp_path: Path) -> Path:
        """Two axes × two severities × ~120 paired recordings per cell.

        Construction:
          * cell A: naive ~ N(label*0.2,1), canon ~ N(label*2.0,1)
            → canon should be hugely better (DeLong p<<0.05)
          * cell B: naive == canon (identical) → DeLong p == 1.0, d == 0

        That gives us a CSV exercising both the "significant" and the
        "null" cases in one go.
        """
        rng = np.random.default_rng(101)
        N = 120
        labels = np.zeros(N, dtype=np.int64)
        labels[: N // 2] = 1
        rng.shuffle(labels)
        path = tmp_path / "per_rec.csv"
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["rec_id", "axis", "severity", "arm",
                         "label", "logit", "prob", "var_logit"])
            for axis, sev_label, sep_naive, sep_canon in [
                ("calibration", "0.5", 0.2, 2.0),
                ("broadband",   "20", 1.0, 1.0),     # identical arms by construction
            ]:
                # Generate per-arm scores (treat the score as both logit + prob
                # for simplicity since the file only cares about probability
                # column for AUROC).
                noise_n = rng.normal(0.0, 1.0, size=N)
                noise_c = noise_n.copy() if axis == "broadband" else rng.normal(0.0, 1.0, size=N)
                scores_n = labels * sep_naive + noise_n
                scores_c = labels * sep_canon + noise_c
                # Squash to probs via sigmoid for realism (monotonic, doesn't
                # change AUROC).
                p_n = 1.0 / (1.0 + np.exp(-scores_n))
                p_c = 1.0 / (1.0 + np.exp(-scores_c))
                for i in range(N):
                    rid = f"rec_{i:03d}"
                    wr.writerow([
                        rid, axis, sev_label, "naive",
                        int(labels[i]), f"{scores_n[i]:.4f}", f"{p_n[i]:.4f}", "0.001",
                    ])
                    wr.writerow([
                        rid, axis, sev_label, "canonicalized",
                        int(labels[i]), f"{scores_c[i]:.4f}", f"{p_c[i]:.4f}", "0.001",
                    ])
        return path

    def test_compute_emits_expected_columns(self, tmp_path):
        path = self._build_synthetic_csv(tmp_path)
        rows_in = load_per_recording_csv(path)
        assert len(rows_in) > 0
        out_rows = compute_statistical_tests(rows_in, n_resamples=200, seed=7)
        # Two (axis, severity) cells → two rows
        assert len(out_rows) == 2
        expected = {
            "axis", "severity", "n_recordings",
            "auroc_naive", "auroc_canon", "auroc_delta",
            "delta_ci_lo", "delta_ci_hi", "p_delong", "cohens_d",
        }
        for r in out_rows:
            assert set(r.keys()) == expected

        # Calibration cell should be highly significant; broadband null.
        by_axis = {r["axis"]: r for r in out_rows}
        assert by_axis["calibration"]["p_delong"] < 0.01
        assert by_axis["calibration"]["auroc_delta"] > 0
        assert by_axis["broadband"]["p_delong"] == pytest.approx(1.0)
        assert by_axis["broadband"]["auroc_delta"] == pytest.approx(0.0, abs=1e-9)
        assert by_axis["broadband"]["cohens_d"] == 0.0

    def test_write_statistical_tests_csv_roundtrip(self, tmp_path):
        path = self._build_synthetic_csv(tmp_path)
        out_csv = tmp_path / "stats.csv"
        out_rows = compute_statistical_tests(
            load_per_recording_csv(path), n_resamples=100, seed=7
        )
        write_statistical_tests_csv(out_rows, out_csv)
        assert out_csv.exists()
        with open(out_csv, "r", newline="") as fh:
            rdr = csv.DictReader(fh)
            csv_rows = list(rdr)
        assert len(csv_rows) == len(out_rows)
        for r in csv_rows:
            assert set(r.keys()) >= {
                "axis", "severity", "n_recordings",
                "auroc_naive", "auroc_canon", "auroc_delta",
                "delta_ci_lo", "delta_ci_hi", "p_delong", "cohens_d",
            }
            # Numeric fields should parse cleanly
            float(r["auroc_naive"])
            float(r["auroc_canon"])
            float(r["auroc_delta"])
            float(r["p_delong"])
            float(r["cohens_d"])
            int(r["n_recordings"])
