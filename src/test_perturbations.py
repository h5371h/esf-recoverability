"""
Unit tests for SPMB 2026 acquisition-shift perturbations.

Each perturbation axis gets:
  * a shape-and-dtype contract test
  * a behaviour test (naive vs canonicalized differ where expected,
    canonicalized restores key statistics where claimed)
  * an edge-case test (e.g. zero amplitude → pass-through)

No real EDFs, no real model — pure numpy + scipy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the src/ package directory is importable.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from perturbations import (  # noqa: E402
    AXES,
    BROADBAND_SNR_DB_GRID,
    CALIBRATION_GAIN_GRID,
    CANONICAL_FS_HZ,
    CANONICAL_N_CHANS,
    MONTAGE_SUBSET_GRID,
    POWER_LINE_AMPLITUDE_UV_GRID,
    SAMPLING_RATE_GRID_HZ,
    broadband_noise,
    calibration_gain,
    montage_change,
    power_line_50hz,
    sampling_rate_mismatch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synth_signal() -> np.ndarray:
    """10-second 19-channel synthetic EEG at canonical 250 Hz.

    Per-channel different alpha-band sinusoid amplitude → channels are
    distinguishable for tests that check per-channel gain.
    """
    rng = np.random.default_rng(42)
    fs = int(CANONICAL_FS_HZ)
    n_samp = 10 * fs
    t = np.arange(n_samp) / fs
    sig = np.zeros((CANONICAL_N_CHANS, n_samp), dtype=np.float32)
    for ch in range(CANONICAL_N_CHANS):
        amp = 20.0 + 5.0 * ch  # 20, 25, 30 ... µV
        sig[ch, :] = amp * np.sin(2 * np.pi * 10.0 * t + ch * 0.1)
        # Add a small noise floor so variance > 0.
        sig[ch, :] += rng.normal(0.0, 1.0, size=n_samp).astype(np.float32)
    return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# Sampling-rate mismatch
# ---------------------------------------------------------------------------
class TestSamplingRateMismatch:
    def test_no_op_when_native_matches_target(self, synth_signal):
        out = sampling_rate_mismatch(
            synth_signal, {"fs_native": 250, "target_fs": 250, "arm": "naive"}
        )
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, synth_signal)

    def test_naive_arm_is_passthrough(self, synth_signal):
        out = sampling_rate_mismatch(
            synth_signal,
            {"fs_native": 128, "target_fs": 250, "arm": "naive"},
        )
        np.testing.assert_array_equal(out, synth_signal)

    def test_canonicalized_resamples_to_target_rate(self, synth_signal):
        # synth_signal is 10 s at "native" 128 Hz interpretation = 1280 samples
        # canonicalized to 250 Hz should be ~2500 samples (10 s × 250 Hz)
        # but we pass synth_signal as if it were 128 Hz — resample_poly will
        # apply up=125, down=64 (gcd = 2 → 250/2=125, 128/2=64).
        out = sampling_rate_mismatch(
            synth_signal,
            {"fs_native": 128, "target_fs": 250, "arm": "canonicalized"},
        )
        assert out.dtype == np.float32
        assert out.shape[0] == CANONICAL_N_CHANS
        # Resample length ≈ n_samp * 250 / 128
        expected_len = synth_signal.shape[1] * 250 // 128
        # resample_poly may add ±a few samples for filter delay; allow 5%
        assert abs(out.shape[1] - expected_len) <= int(0.05 * expected_len) + 2

    def test_canonicalize_higher_rate_to_lower(self, synth_signal):
        out = sampling_rate_mismatch(
            synth_signal,
            {"fs_native": 512, "target_fs": 250, "arm": "canonicalized"},
        )
        assert out.shape[0] == CANONICAL_N_CHANS
        # Resampling 512 → 250 should produce ~250/512 of the original samples
        assert out.shape[1] < synth_signal.shape[1]


# ---------------------------------------------------------------------------
# Calibration gain
# ---------------------------------------------------------------------------
class TestCalibrationGain:
    def test_scalar_gain_naive_arm_multiplies(self, synth_signal):
        out = calibration_gain(synth_signal, {"gain": 2.0, "arm": "naive"})
        np.testing.assert_allclose(out, synth_signal * 2.0, rtol=1e-5)

    def test_scalar_gain_canonicalized_arm_divides(self, synth_signal):
        out = calibration_gain(synth_signal, {"gain": 2.0, "arm": "canonicalized"})
        np.testing.assert_allclose(out, synth_signal / 2.0, rtol=1e-5)

    def test_canonical_recovers_naive_perturbation_round_trip(self, synth_signal):
        """The keystone-axis property: canonicalize(naive(x, g), g) ≈ x.

        Holds because both arms are deterministic and use the same per-channel
        gain (in deployment this is the device-calibration metadata).
        """
        gain = 1.5
        perturbed = calibration_gain(synth_signal, {"gain": gain, "arm": "naive"})
        recovered = calibration_gain(perturbed, {"gain": gain, "arm": "canonicalized"})
        np.testing.assert_allclose(recovered, synth_signal, rtol=1e-5, atol=1e-5)

    def test_per_channel_gain_vector(self, synth_signal):
        gain_vec = np.linspace(0.5, 2.0, CANONICAL_N_CHANS).astype(np.float32)
        out = calibration_gain(synth_signal, {"gain": gain_vec, "arm": "naive"})
        for ch in range(CANONICAL_N_CHANS):
            np.testing.assert_allclose(
                out[ch], synth_signal[ch] * gain_vec[ch], rtol=1e-5
            )

    def test_zero_gain_raises(self, synth_signal):
        with pytest.raises(ValueError):
            calibration_gain(synth_signal, {"gain": 0.0, "arm": "naive"})


# ---------------------------------------------------------------------------
# Power-line 50 Hz
# ---------------------------------------------------------------------------
class TestPowerLine50Hz:
    def test_zero_amplitude_passes_through(self, synth_signal):
        out = power_line_50hz(synth_signal, {"amplitude_uv": 0.0, "arm": "naive"})
        np.testing.assert_array_equal(out, synth_signal)

    def test_naive_arm_injects_50hz_energy(self, synth_signal):
        amp = 30.0
        out = power_line_50hz(
            synth_signal,
            {"amplitude_uv": amp, "harmonics": 1, "fs": CANONICAL_FS_HZ, "arm": "naive"},
        )
        # 50 Hz spectral component should jump.
        from numpy.fft import rfft, rfftfreq

        freqs = rfftfreq(synth_signal.shape[1], d=1.0 / CANONICAL_FS_HZ)
        idx_50 = int(np.argmin(np.abs(freqs - 50.0)))
        before_50 = float(np.mean(np.abs(rfft(synth_signal, axis=1))[:, idx_50]))
        after_50 = float(np.mean(np.abs(rfft(out, axis=1))[:, idx_50]))
        assert after_50 > 5 * before_50, (
            f"Naive arm should boost 50 Hz energy ≥5x; got {after_50/before_50:.2f}x"
        )

    def test_canonicalized_arm_notches_50hz(self, synth_signal):
        amp = 30.0
        contaminated = power_line_50hz(
            synth_signal,
            {"amplitude_uv": amp, "harmonics": 3, "fs": CANONICAL_FS_HZ, "arm": "naive"},
        )
        cleaned = power_line_50hz(
            synth_signal,
            {"amplitude_uv": amp, "harmonics": 3, "fs": CANONICAL_FS_HZ, "arm": "canonicalized"},
        )
        from numpy.fft import rfft, rfftfreq

        freqs = rfftfreq(synth_signal.shape[1], d=1.0 / CANONICAL_FS_HZ)
        idx_50 = int(np.argmin(np.abs(freqs - 50.0)))
        contam_e = float(np.mean(np.abs(rfft(contaminated, axis=1))[:, idx_50]))
        cleaned_e = float(np.mean(np.abs(rfft(cleaned, axis=1))[:, idx_50]))
        # Canonical notch should kill at least 80% of the 50 Hz energy.
        assert cleaned_e < 0.2 * contam_e, (
            f"Canonical arm notch insufficient: {cleaned_e:.3f} / {contam_e:.3f}"
        )

    def test_dtype_is_float32(self, synth_signal):
        out = power_line_50hz(
            synth_signal,
            {"amplitude_uv": 15.0, "harmonics": 2, "arm": "canonicalized"},
        )
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Montage change
# ---------------------------------------------------------------------------
class TestMontageChange:
    def test_full19_is_passthrough(self, synth_signal):
        out = montage_change(synth_signal, {"subset": "full19", "arm": "naive"})
        np.testing.assert_array_equal(out, synth_signal)

    def test_legacy16_naive_zeroes_dropped_channels(self, synth_signal):
        from perturbations import (
            CANONICAL_CHANNEL_ORDER,
        )

        out = montage_change(synth_signal, {"subset": "legacy16", "arm": "naive"})
        drop_idx = [CANONICAL_CHANNEL_ORDER.index(c) for c in ("F7", "F8", "T6")]
        for i in drop_idx:
            assert np.all(out[i] == 0.0), f"channel {i} not zeroed in naive legacy16"
        # All other channels untouched.
        keep_mask = np.ones(CANONICAL_N_CHANS, dtype=bool)
        keep_mask[drop_idx] = False
        np.testing.assert_array_equal(out[keep_mask], synth_signal[keep_mask])

    def test_legacy16_canonicalized_fills_dropped_channels_nonzero(self, synth_signal):
        from perturbations import (
            CANONICAL_CHANNEL_ORDER,
        )

        out = montage_change(
            synth_signal, {"subset": "legacy16", "arm": "canonicalized"}
        )
        drop_idx = [CANONICAL_CHANNEL_ORDER.index(c) for c in ("F7", "F8", "T6")]
        for i in drop_idx:
            assert not np.all(out[i] == 0.0), (
                f"channel {i} should be filled (median across remaining), not zeroed"
            )

    def test_bipolar18_naive_returns_19_channels(self, synth_signal):
        out = montage_change(synth_signal, {"subset": "bipolar18", "arm": "naive"})
        assert out.shape == synth_signal.shape

    def test_bipolar18_canonicalized_recovers_approximate_monopolar(self, synth_signal):
        # The pseudoinverse reconstruction is underdetermined (18 bipolar
        # equations for 19 monopolar unknowns — the common-mode is in the
        # null space). Posterior/central channels recover well; the rest
        # show only the differential structure. We test that the
        # canonical arm meaningfully recovers a substantial subset rather
        # than producing zeros or random noise.
        out = montage_change(
            synth_signal, {"subset": "bipolar18", "arm": "canonicalized"}
        )
        corrs = []
        for ch in range(CANONICAL_N_CHANS):
            if np.std(out[ch]) < 1e-9 or np.std(synth_signal[ch]) < 1e-9:
                continue
            corrs.append(float(np.corrcoef(out[ch], synth_signal[ch])[0, 1]))
        # Posterior + central channels (the well-conditioned ones — at
        # least 8 of 19 in the double-banana montage) should recover
        # strongly (|corr| > 0.7). This is the headline property of the
        # canonicalized arm vs naive zero-projection.
        strong = sum(1 for c in corrs if abs(c) > 0.7)
        assert strong >= 6, (
            f"Bipolar reconstruction failed to recover ≥6 channels strongly: "
            f"|corr| values = {[round(c, 2) for c in corrs]}"
        )


# ---------------------------------------------------------------------------
# Broadband noise
# ---------------------------------------------------------------------------
class TestBroadbandNoise:
    def test_high_snr_changes_signal_little(self, synth_signal):
        out = broadband_noise(synth_signal, {"snr_db": 40.0, "arm": "naive"})
        rmse = float(np.sqrt(np.mean((out - synth_signal) ** 2)))
        sig_rms = float(np.sqrt(np.mean(synth_signal ** 2)))
        # At 40 dB SNR the per-sample RMSE should be < 10% of signal RMS.
        assert rmse < 0.1 * sig_rms, (
            f"40dB SNR introduced too much noise: rmse={rmse:.3f} sigrms={sig_rms:.3f}"
        )

    def test_low_snr_dominates_signal(self, synth_signal):
        out = broadband_noise(synth_signal, {"snr_db": -10.0, "arm": "naive"})
        rmse = float(np.sqrt(np.mean((out - synth_signal) ** 2)))
        sig_rms = float(np.sqrt(np.mean(synth_signal ** 2)))
        assert rmse > sig_rms, (
            f"-10dB SNR should make noise > signal: rmse={rmse:.3f} sigrms={sig_rms:.3f}"
        )

    def test_naive_and_canonicalized_arms_identical(self, synth_signal):
        # Floor-condition by construction: both arms apply same noise.
        naive_out = broadband_noise(
            synth_signal, {"snr_db": 5.0, "arm": "naive", "seed": 11}
        )
        canon_out = broadband_noise(
            synth_signal, {"snr_db": 5.0, "arm": "canonicalized", "seed": 11}
        )
        np.testing.assert_array_equal(naive_out, canon_out)

    def test_seed_determinism(self, synth_signal):
        a = broadband_noise(synth_signal, {"snr_db": 5.0, "arm": "naive", "seed": 99})
        b = broadband_noise(synth_signal, {"snr_db": 5.0, "arm": "naive", "seed": 99})
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Axis registry
# ---------------------------------------------------------------------------
class TestAxisRegistry:
    def test_all_five_axes_present(self):
        assert set(AXES.keys()) == {
            "sampling_rate", "calibration", "power_line", "montage", "broadband"
        }

    def test_axis_specs_have_required_keys(self):
        for name, spec in AXES.items():
            assert callable(spec["fn"]), f"{name} fn is not callable"
            assert spec["severities"], f"{name} has empty severity list"
            assert spec["severity_key"], f"{name} has no severity_key"
            for sev in spec["severities"]:
                assert spec["severity_key"] in sev, (
                    f"{name} severity {sev} missing key {spec['severity_key']!r}"
                )

    @pytest.mark.parametrize("axis", ["sampling_rate", "calibration",
                                     "power_line", "montage", "broadband"])
    def test_axis_runs_end_to_end_synthetic(self, axis, synth_signal):
        spec = AXES[axis]
        for sev in spec["severities"]:
            for arm in ("naive", "canonicalized"):
                params = dict(sev)
                params["arm"] = arm
                out = spec["fn"](synth_signal, params)
                assert out.dtype == np.float32, (
                    f"{axis}/{sev}/{arm} produced dtype {out.dtype}"
                )
                assert out.shape[0] == CANONICAL_N_CHANS, (
                    f"{axis}/{sev}/{arm} channel count wrong: {out.shape}"
                )
                assert out.shape[1] > 0, (
                    f"{axis}/{sev}/{arm} produced zero-length output"
                )
                assert np.all(np.isfinite(out)), (
                    f"{axis}/{sev}/{arm} produced non-finite values"
                )
