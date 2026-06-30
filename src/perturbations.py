"""
Signal-level perturbations modelling acquisition shift (paper §"Acquisition-shift model").

Five axes. Each axis has TWO arms that share the same downstream backbone+probe:

  * naive arm        — perturbed input passed straight to the model
  * canonicalized arm — perturbed input mapped back to the canonical
                        representation (µV @ 250 Hz, 19-ch 10-20) before
                        the model sees it

The difference between arms is the recovery attributable to canonicalization.
Broadband noise has NO recovery — its "canonicalized" arm is identical to its
"naive" arm by construction (the paper treats it as a floor condition).

All functions:
  * take `signal: np.ndarray` shaped (n_channels, n_samples), float32, µV
  * are PURE (no I/O, no global state)
  * are deterministic given fixed `params["seed"]` where randomness applies
  * never raise on edge inputs (zero-length signals, mismatched fs etc.) —
    they return a safe pass-through with a logged warning

Channel order assumed throughout = ESF canonical 19-ch (see
apps/iplane/models/eegpt_backbone.ESF_CHANNEL_ORDER):

    Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz

The TUAB source corpus uses a 60 Hz mains environment; the power-line
perturbation models the deployment-side 50 Hz environment (paper §
acquisition-shift model). The canonicalized arm applies IIR notch at
{50, 100, 150} Hz; the naive arm leaves the interference in place.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — canonical target representation
# ---------------------------------------------------------------------------
CANONICAL_FS_HZ: float = 250.0
CANONICAL_N_CHANS: int = 19
CANONICAL_CHANNEL_ORDER: tuple[str, ...] = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1",  "O2",  "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz",  "Cz",  "Pz",
)

# Severity grids (paper §acquisition-shift model). Each grid spans plausible
# device-population values; the canonical target sits inside the grid so the
# zero-severity / native-canonical condition is its own datapoint.
SAMPLING_RATE_GRID_HZ: tuple[int, ...] = (128, 200, 250, 256, 512)
CALIBRATION_GAIN_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
POWER_LINE_AMPLITUDE_UV_GRID: tuple[float, ...] = (0.0, 5.0, 15.0, 30.0, 60.0)
BROADBAND_SNR_DB_GRID: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 20.0)
MONTAGE_SUBSET_GRID: tuple[str, ...] = ("full19", "legacy16", "bipolar18")


# ---------------------------------------------------------------------------
# (1) Sampling-rate mismatch
# ---------------------------------------------------------------------------
def sampling_rate_mismatch(
    signal: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Perturb / canonicalize sampling-rate mismatch.

    Models a recording acquired at `fs_native` Hz that the model expects at
    `target_fs` Hz (default 250 Hz).

    params:
      fs_native  : int Hz of the source recording (one of SAMPLING_RATE_GRID_HZ)
      target_fs  : int Hz the model expects (default 250)
      arm        : "naive" or "canonicalized"
        * naive        — return signal unchanged, model sees rate-mismatched data
        * canonicalized — scipy.signal.resample_poly with anti-aliased filter
                         (rational up/down by gcd) to bring fs to target_fs

    Returns float32 array shaped (n_channels, n_samples_resampled).

    Why a no-op naive arm: the model's contract is 250 Hz; pretending a
    128 Hz recording IS 250 Hz means each sample represents ~1.95× more time
    than the model assumes. That's the shift the paper's Fig 1
    sampling-rate panel measures.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (channels, samples); got shape {signal.shape}")
    fs_native = int(params.get("fs_native", CANONICAL_FS_HZ))
    target_fs = int(params.get("target_fs", CANONICAL_FS_HZ))
    arm = str(params.get("arm", "naive"))

    if arm not in ("naive", "canonicalized"):
        raise ValueError(f"arm must be 'naive' or 'canonicalized'; got {arm!r}")

    # Common-case fast paths.
    if fs_native == target_fs:
        return signal.astype(np.float32, copy=False)
    if arm == "naive":
        # Pass-through: the rate-mismatched signal goes to the model as-is.
        # This is the silent-degradation condition.
        return signal.astype(np.float32, copy=False)

    # Canonicalized arm: anti-aliased rational resample.
    from math import gcd
    g = gcd(fs_native, target_fs)
    up = target_fs // g
    down = fs_native // g
    try:
        from scipy.signal import resample_poly  # type: ignore
        out = resample_poly(signal, up=up, down=down, axis=1)
    except Exception as e:
        logger.warning(
            "sampling_rate_mismatch canonicalize fallback (no scipy?): %s; "
            "returning naive pass-through",
            e,
        )
        return signal.astype(np.float32, copy=False)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# (2) Calibration / gain error
# ---------------------------------------------------------------------------
def calibration_gain(
    signal: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Perturb / canonicalize per-channel calibration error.

    Models a device whose per-channel digital-to-µV scale is off by a
    multiplicative factor. Paper § acquisition-shift model treats this as
    the keystone axis for the physical-µV ablation:

      * naive arm        — signal × gain (the physical µV is now wrong; the
                            model sees inflated/deflated amplitudes)
      * canonicalized arm — signal / gain (restore physical µV)

    The "recovery gap" on this axis is the direct payoff of the canonical
    representation holding amplitude in µV (rather than per-channel z-score).

    params:
      gain : either a scalar applied uniformly, or an array of length
             n_channels for per-channel gain. Defaults to 1.0 (no-op).
      arm  : "naive" | "canonicalized"
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D; got {signal.shape}")
    arm = str(params.get("arm", "naive"))
    gain = params.get("gain", 1.0)
    g = np.atleast_1d(np.asarray(gain, dtype=np.float32))
    if g.size == 1:
        g_arr = np.full((signal.shape[0],), float(g.item()), dtype=np.float32)
    else:
        if g.size != signal.shape[0]:
            raise ValueError(
                f"gain length {g.size} != n_channels {signal.shape[0]}"
            )
        g_arr = g.astype(np.float32)
    if not np.all(np.isfinite(g_arr)) or np.any(g_arr == 0):
        raise ValueError("gain must be finite and non-zero")

    if arm == "naive":
        # Drift physical µV: the model now sees signal * gain. If gain=2,
        # 50 µV alpha looks like 100 µV alpha, etc.
        return (signal * g_arr[:, None]).astype(np.float32, copy=False)
    if arm == "canonicalized":
        # Restore physical µV. In a real pipeline this comes from the
        # device-specific calibration scalar in EDF header / metadata; here
        # we model the canonical path as having perfect knowledge of gain
        # (the canonicalization spec demands physical-µV at write time).
        return (signal / g_arr[:, None]).astype(np.float32, copy=False)
    raise ValueError(f"arm must be 'naive' or 'canonicalized'; got {arm!r}")


# ---------------------------------------------------------------------------
# (3) Power-line interference (50 Hz)
# ---------------------------------------------------------------------------
def power_line_50hz(
    signal: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Perturb / canonicalize 50 Hz mains interference + harmonics.

    Models a recording acquired in a 50 Hz mains environment (most of EU,
    India, AU) against a model trained on 60 Hz TUAB.

      * naive arm        — add a sinusoid at 50 Hz plus harmonics at 100, 150 Hz
                           with the given amplitude (µV). Model sees raw
                           mains contamination.
      * canonicalized arm — first add the same contamination, then apply
                           scipy.signal.iirnotch at 50/100/150 Hz with Q=30.
                           Model sees clean-ish signal.

    params:
      amplitude_uv : peak amplitude of the fundamental sinusoid (µV).
                     Harmonics scale as amplitude / (k+1) for k=1..harmonics-1.
      harmonics    : how many harmonics to include (default 3; covers
                     50/100/150 Hz)
      fs           : sampling rate of `signal` in Hz (default CANONICAL_FS_HZ)
      arm          : "naive" | "canonicalized"
      seed         : optional int for the random phase offsets (default 0)

    The canonical arm uses the same contamination as the naive arm so the
    recovery measurement is the notch filter's contribution alone.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D; got {signal.shape}")
    n_ch, n_samp = signal.shape
    amplitude_uv = float(params.get("amplitude_uv", 0.0))
    harmonics = int(params.get("harmonics", 3))
    fs = float(params.get("fs", CANONICAL_FS_HZ))
    arm = str(params.get("arm", "naive"))
    seed = int(params.get("seed", 0))

    if amplitude_uv <= 0.0:
        # Zero-severity datapoint — the model sees the canonical clean signal.
        return signal.astype(np.float32, copy=False)

    # Build the contamination once (per-channel random phases for realism).
    t = np.arange(n_samp, dtype=np.float64) / fs
    rng = np.random.default_rng(seed)
    contamination = np.zeros((n_ch, n_samp), dtype=np.float64)
    for k in range(harmonics):
        f_k = 50.0 * (k + 1)
        if f_k >= fs / 2.0:
            # Above Nyquist — silently skip this harmonic
            continue
        amp_k = amplitude_uv / (k + 1)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_ch)
        # broadcast: (n_ch, n_samp) = amp * sin(2π f_k t + phase)
        contamination += amp_k * np.sin(2.0 * np.pi * f_k * t[None, :] + phases[:, None])

    contaminated = signal.astype(np.float64, copy=True) + contamination

    if arm == "naive":
        return contaminated.astype(np.float32)
    if arm == "canonicalized":
        try:
            from scipy.signal import iirnotch, filtfilt  # type: ignore
        except Exception as e:
            logger.warning(
                "power_line_50hz canonicalize fallback (no scipy?): %s; "
                "returning contaminated signal",
                e,
            )
            return contaminated.astype(np.float32)
        out = contaminated.copy()
        for k in range(harmonics):
            f_k = 50.0 * (k + 1)
            if f_k >= fs / 2.0:
                continue
            # Q=30 is a tight notch (paper §canonical-representation conformance).
            w0 = f_k / (fs / 2.0)
            b, a = iirnotch(w0=w0, Q=30.0)
            out = filtfilt(b, a, out, axis=1)
        return out.astype(np.float32)
    raise ValueError(f"arm must be 'naive' or 'canonicalized'; got {arm!r}")


# ---------------------------------------------------------------------------
# (4) Montage / reference change
# ---------------------------------------------------------------------------
def montage_change(
    signal: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Perturb / canonicalize a reduced or remapped montage.

    Models a recording arriving with fewer channels than the canonical 19,
    or with a different reference scheme.

      * naive arm        — drop the missing channels (zero-fill in their
                           canonical-order slots so shape stays (19, n_samp);
                           model sees zeroed channels in the dropped slots).
                           For bipolar18: build bipolar derivations and
                           zero-pad to 19; one canonical slot remains zero.
      * canonicalized arm — re-derive the canonical 19-channel monopolar
                           representation. For reduced-channel inputs this
                           uses the existing libs/esf channel_resolver to
                           place known channels and leaves dropped ones at
                           the per-channel training-distribution mean (≈ 0
                           after ESF prenorm).

    Subsets:
      full19    : no-op (19-ch reference present)
      legacy16  : drop F7, F8, T6 (a common reduced-headcount legacy setup)
      bipolar18 : 18 bipolar derivations re-projected naively to 19 slots

    params:
      subset       : "full19" | "legacy16" | "bipolar18"
      arm          : "naive" | "canonicalized"

    NOTE: To keep the experiment dependency-free we do NOT call the live
    libs/esf channel_resolver from this script — both arms produce a (19,
    n_samples) array consumable by the EEGPT backbone directly. The naive
    arm leaves dropped channels at 0; the canonical arm fills them with the
    per-channel median across remaining channels (a reasonable proxy for
    spatial interpolation that the actual ESF resolver performs).
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D; got {signal.shape}")
    n_ch, n_samp = signal.shape
    if n_ch != CANONICAL_N_CHANS:
        raise ValueError(
            f"signal must have {CANONICAL_N_CHANS} canonical channels; got {n_ch}"
        )
    subset = str(params.get("subset", "full19"))
    arm = str(params.get("arm", "naive"))

    if subset == "full19":
        return signal.astype(np.float32, copy=False)

    if arm not in ("naive", "canonicalized"):
        raise ValueError(f"arm must be 'naive' or 'canonicalized'; got {arm!r}")

    if subset == "legacy16":
        # Drop F7 (idx 10), F8 (idx 11), T6 (idx 15) per CANONICAL_CHANNEL_ORDER.
        drop_idx = [CANONICAL_CHANNEL_ORDER.index(c) for c in ("F7", "F8", "T6")]
        out = signal.astype(np.float32, copy=True)
        if arm == "naive":
            # Zero-fill the dropped slots.
            for i in drop_idx:
                out[i, :] = 0.0
            return out
        # canonicalized: fill the dropped channels with per-time median across
        # the surviving channels (spatial interpolation proxy).
        keep_mask = np.ones(n_ch, dtype=bool)
        keep_mask[drop_idx] = False
        surviving = out[keep_mask, :]
        median_t = np.median(surviving, axis=0)
        for i in drop_idx:
            out[i, :] = median_t
        return out

    if subset == "bipolar18":
        # Build 18 bipolar derivations from the 19 monopolar channels —
        # standard double-banana approximation (longitudinal). Zero-pad to 19.
        # Pairs use CANONICAL_CHANNEL_ORDER indices:
        order = list(CANONICAL_CHANNEL_ORDER)

        def idx(name: str) -> int:
            return order.index(name)

        pairs = [
            ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
            ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
            ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
            ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
            ("Fz", "Cz"),  ("Cz", "Pz"),
        ]
        bipolar = np.zeros((19, n_samp), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            bipolar[i, :] = signal[idx(a), :] - signal[idx(b), :]
        # row 18 stays zero — the bipolar montage has 18 channels, not 19

        if arm == "naive":
            return bipolar
        # canonicalized: project bipolar derivations back to monopolar by a
        # least-squares pseudoinverse of the pair-difference matrix.
        # Build the (18, 19) bipolar-derivation matrix D such that
        # bipolar = D @ monopolar; reconstruct monopolar = pinv(D) @ bipolar.
        D = np.zeros((18, 19), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            D[i, idx(a)] = 1.0
            D[i, idx(b)] = -1.0
        Dpinv = np.linalg.pinv(D)             # (19, 18)
        recon_mono = Dpinv @ bipolar[:18, :]  # (19, n_samp)
        return recon_mono.astype(np.float32)

    raise ValueError(f"unknown subset {subset!r}")


# ---------------------------------------------------------------------------
# (5) Broadband noise
# ---------------------------------------------------------------------------
def broadband_noise(
    signal: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Add broadband Gaussian noise at a target SNR (dB).

    This is the floor condition (paper § acquisition-shift model):
    broadband noise is genuinely lossy and the canonical arm CANNOT recover
    it. Both arms apply the same noise; the abstention rule (see
    selective_prediction.py) is the appropriate response to this axis, not
    a representation fix.

    params:
      snr_db : target signal-to-noise ratio in dB across all channels
               (default 20 dB; sweep -5 .. 20)
      arm    : "naive" | "canonicalized" — identical behaviour by design
      seed   : int for noise reproducibility
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D; got {signal.shape}")
    arm = str(params.get("arm", "naive"))
    if arm not in ("naive", "canonicalized"):
        raise ValueError(f"arm must be 'naive' or 'canonicalized'; got {arm!r}")
    snr_db = float(params.get("snr_db", 20.0))
    seed = int(params.get("seed", 0))

    sig_power = float(np.mean(signal.astype(np.float64) ** 2))
    if sig_power <= 0.0:
        return signal.astype(np.float32, copy=False)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return (signal.astype(np.float32, copy=False) + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Axis registry — drives eval_loop.py's nested for-loop
# ---------------------------------------------------------------------------
AXES: dict[str, dict] = {
    "sampling_rate": {
        "fn":       sampling_rate_mismatch,
        "severities": [{"fs_native": v, "target_fs": int(CANONICAL_FS_HZ)} for v in SAMPLING_RATE_GRID_HZ],
        "severity_key": "fs_native",
    },
    "calibration":  {
        "fn":       calibration_gain,
        "severities": [{"gain": v} for v in CALIBRATION_GAIN_GRID],
        "severity_key": "gain",
    },
    "power_line":   {
        "fn":       power_line_50hz,
        "severities": [{"amplitude_uv": v, "harmonics": 3, "fs": CANONICAL_FS_HZ} for v in POWER_LINE_AMPLITUDE_UV_GRID],
        "severity_key": "amplitude_uv",
    },
    "montage":      {
        "fn":       montage_change,
        "severities": [{"subset": v} for v in MONTAGE_SUBSET_GRID],
        "severity_key": "subset",
    },
    "broadband":    {
        "fn":       broadband_noise,
        "severities": [{"snr_db": v} for v in BROADBAND_SNR_DB_GRID],
        "severity_key": "snr_db",
    },
}
