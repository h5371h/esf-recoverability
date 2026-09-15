"""Acquisition operators A_theta acting on the RAW microvolt recording at its native rate,
i.e. upstream of canonicalisation. Each returns (sig_uv, labels, fs, line_freq)."""
from __future__ import annotations
from math import gcd
import numpy as np
from scipy import signal as sp

def sampling_rate(sig, labels, fs, lf, fs_native: int, rng=None):
    if abs(fs - fs_native) < 0.5: return sig, labels, fs, lf
    g = gcd(int(round(fs)), int(fs_native)); up, down = int(fs_native)//g, int(round(fs))//g
    out = sp.resample_poly(sig.astype(np.float64), up=up, down=down, axis=1).astype(np.float32)  # anti-aliased
    return out, labels, float(fs_native), lf

def gain(sig, labels, fs, lf, g: float, rng=None):
    return (sig * np.float32(g)), labels, fs, lf

def _eeg_rows(labels):
    from .esf.channels import resolve_channel_name
    return [i for i, l in enumerate(labels) if resolve_channel_name(l) is not None]

def power_line(sig, labels, fs, lf, amp_uv: float, rng=None):
    """Additive mains interference at the regional mains frequency plus 2nd/3rd harmonics (1/2, 1/3 amplitude).
    Peak amplitude amp_uv (microvolts) is the across-electrode mean; each EEG electrode gets its own amplitude
    factor (log-normal, sigma 0.3) and phase (uniform), as real pickup differs per electrode/lead. Non-EEG rows untouched."""
    if amp_uv <= 0: return sig, labels, fs, lf
    rng = rng or np.random.default_rng(0)
    rows = _eeg_rows(labels); t = np.arange(sig.shape[1], dtype=np.float64) / fs
    out = sig.copy()
    for i in rows:
        a = amp_uv * float(np.exp(rng.normal(0.0, 0.3))); ph = rng.uniform(0, 2*np.pi, size=3)
        n = np.zeros_like(t)
        for k in (1, 2, 3):                       # harmonics only below the native Nyquist (anti-alias filter)
            if k * lf < fs / 2: n += (a / k) * np.sin(2*np.pi*k*lf*t + ph[k-1])
        out[i] = sig[i] + n.astype(np.float32)
    return out, labels, fs, lf

LEGACY16_DROP = {"T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"}
def montage(sig, labels, fs, lf, config: str, rng=None):
    from .esf.channels import resolve_channel_name
    if config == "full19": return sig, labels, fs, lf
    if config == "legacy16":
        keep = [i for i, l in enumerate(labels) if (resolve_channel_name(l) or "").upper() not in LEGACY16_DROP]
        return sig[keep], [labels[i] for i in keep], fs, lf
    if config == "bipolar18":
        # Longitudinal double-banana derivations from the referential raw, then the v1 pseudo-inverse:
        # each derivation A-B is assigned back to slot A (anode) — the naive re-projection used in v1.
        pairs = [("FP1","F7"),("F7","T3"),("T3","T5"),("T5","O1"),("FP2","F8"),("F8","T4"),("T4","T6"),("T6","O2"),
                 ("FP1","F3"),("F3","C3"),("C3","P3"),("P3","O1"),("FP2","F4"),("F4","C4"),("C4","P4"),("P4","O2"),
                 ("FZ","CZ"),("CZ","PZ")]
        idx = {}
        for i, l in enumerate(labels):
            c = resolve_channel_name(l)
            if c: idx.setdefault(c.upper(), i)
        out, labs = [], []
        for a, b in pairs:
            if a in idx and b in idx and a not in labs:
                out.append(sig[idx[a]] - sig[idx[b]]); labs.append(a)
        return np.stack(out).astype(np.float32), [f"EEG {l}-REF" for l in labs], fs, lf
    raise ValueError(config)

def broadband(sig, labels, fs, lf, snr_db: float, rng=None):
    """Additive white Gaussian noise at a per-channel SNR (dB) relative to each EEG channel's own power."""
    rng = rng or np.random.default_rng(0)
    rows = _eeg_rows(labels); out = sig.copy()
    for i in rows:
        p_sig = float(np.mean(sig[i].astype(np.float64)**2)); p_noise = p_sig / (10**(snr_db/10))
        out[i] = sig[i] + rng.normal(0.0, np.sqrt(p_noise), size=sig.shape[1]).astype(np.float32)
    return out, labels, fs, lf


def lowpass(sig, labels, fs, lf, cutoff_hz: float, rng=None):
    """Effective-bandwidth probe: zero-phase 8th-order Butterworth low-pass at cutoff_hz on the raw EEG rows at native
    rate (cutoff below the native Nyquist). Not an acquisition shift; a measurement of which band the classifier uses.
    No corrective stage exists, so both arms are identical."""
    if cutoff_hz is None or cutoff_hz >= fs / 2: return sig, labels, fs, lf
    rows = _eeg_rows(labels); out = sig.copy()
    sos = sp.butter(8, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    out[rows] = sp.sosfiltfilt(sos, sig[rows].astype(np.float64), axis=1).astype(np.float32)
    return out, labels, fs, lf

AXES = {
  "sampling_rate": {"fn": sampling_rate, "severities": [128, 200, 250, 256, 512], "key": "fs_native", "fmt": lambda v: str(int(v))},
  "calibration":   {"fn": gain,          "severities": [0.5, 0.75, 1.0, 1.5, 2.0], "key": "g",         "fmt": lambda v: str(float(v))},
  "power_line":    {"fn": power_line,    "severities": [0.0, 5.0, 15.0, 30.0, 60.0], "key": "amp_uv",  "fmt": lambda v: str(float(v))},
  "montage":       {"fn": montage,       "severities": ["full19", "legacy16", "bipolar18"], "key": "config", "fmt": str},
  "broadband":     {"fn": broadband,     "severities": [-5.0, 0.0, 5.0, 10.0, 20.0], "key": "snr_db",  "fmt": lambda v: str(float(v))},
  "bandwidth":     {"fn": lowpass,       "severities": [15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 64.0, 80.0, 100.0], "key": "cutoff_hz", "fmt": lambda v: str(float(v))},
}
NAIVE_SWITCH = {  # the ONE corrective stage disabled in the naive arm
  "sampling_rate": dict(do_resample=False),
  "calibration":   dict(norm="fixed"),
  "power_line":    dict(do_notch=False),
  "montage":       dict(car_mode="all"),
  "broadband":     dict(),
  "bandwidth":     dict(),
}
