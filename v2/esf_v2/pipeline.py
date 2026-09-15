"""ESF pipeline with per-stage switches, composed from the released src/esf functions.
canonical():  full ESF exactly as convert.to_esf (map -> resample -> notch -> HP -> CAR -> robust-z)
naive(...):   the same pipeline with ONE corrective stage disabled, per axis:
   sampling_rate : no resample (native samples are treated as 250 Hz)
   calibration   : robust-z replaced by FIXED per-channel scale constants (training-set medians/IQRs)
   power_line    : notch disabled
   montage       : CAR computed over all 19 slots with missing channels as zeros (instead of good-only)
   broadband     : no corrective stage exists; naive == canonical by design
"""
from __future__ import annotations
import warnings, numpy as np
from scipy import signal as sp
from .esf.channels import CH_INDEX, N_CHANNELS, resolve_channel_name
from .esf.convert import _resample, _notch_filter, _common_average_ref
from .esf.normalize import robust_zscore
ESF_SRATE = 250.0

def map_channels(sig_uv: np.ndarray, labels: list[str]) -> np.ndarray:
    esf = np.full((N_CHANNELS, sig_uv.shape[1]), np.nan, np.float32); seen = set()
    for i, lab in enumerate(labels):
        c = resolve_channel_name(lab)
        if c is None: continue
        k = CH_INDEX[c]
        if k in seen: continue
        esf[k] = sig_uv[i]; seen.add(k)
    return esf

def run(sig_uv: np.ndarray, labels: list[str], fs: float, line_freq: float = 60.0, *,
        do_resample=True, do_notch=True, do_car=True, car_mode="good", norm="robust",
        fixed_scale: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    x = map_channels(sig_uv, labels)
    good = ~np.all(np.isnan(x), axis=1)
    if do_resample and abs(fs - ESF_SRATE) > 0.5:
        n_out = max(1, int(round(x.shape[1] * ESF_SRATE / fs)))
        y = np.full((N_CHANNELS, n_out), np.nan, np.float32)
        gi = np.where(good)[0]
        if gi.size:
            r = _resample(x[gi], fs, ESF_SRATE); n = min(r.shape[1], n_out); y[gi, :n] = r[:, :n]
        x = y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if do_notch and good.any():
            x[good] = _notch_filter(x[good], ESF_SRATE, float(line_freq))
        if do_car and good.any():
            sos = sp.butter(4, 0.5, btype="highpass", fs=ESF_SRATE, output="sos")
            x[good] = sp.sosfiltfilt(sos, x[good], axis=1).astype(np.float32)
            if car_mode == "good":
                if good.sum() >= 2: x = _common_average_ref(x, good)
            else:  # naive: average over all 19 slots with missing channels as zeros
                z = np.nan_to_num(x, nan=0.0); x = (z - z.mean(axis=0, keepdims=True)).astype(np.float32)
                x[~good] = np.nan
    if norm == "robust":
        if good.any():
            xn, _, _ = robust_zscore(x[good], sfreq=ESF_SRATE); x[good] = xn
    elif norm == "fixed":
        med, iqr = fixed_scale
        x[good] = ((x[good] - med[good, None]) / np.where(iqr[good, None] > 1e-9, iqr[good, None], 1.0)).astype(np.float32)
    else:
        raise ValueError(norm)
    return np.nan_to_num(x, nan=0.0).astype(np.float32)

def canonical(sig_uv, labels, fs, line_freq=60.0):
    return run(sig_uv, labels, fs, line_freq)

def robust_constants(sig_uv, labels, fs, line_freq=60.0):
    """Per-channel (median, IQR) after the full pre-normalisation pipeline; used to build FIXED scale."""
    x = map_channels(sig_uv, labels); good = ~np.all(np.isnan(x), axis=1)
    if abs(fs - ESF_SRATE) > 0.5:
        gi = np.where(good)[0]; r = _resample(x[gi], fs, ESF_SRATE)
        y = np.full((N_CHANNELS, r.shape[1]), np.nan, np.float32); y[gi] = r; x = y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x[good] = _notch_filter(x[good], ESF_SRATE, float(line_freq))
        sos = sp.butter(4, 0.5, btype="highpass", fs=ESF_SRATE, output="sos")
        x[good] = sp.sosfiltfilt(sos, x[good], axis=1).astype(np.float32)
        if good.sum() >= 2: x = _common_average_ref(x, good)
    _, med, iqr = robust_zscore(x[good], sfreq=ESF_SRATE)
    m = np.full(N_CHANNELS, np.nan); q = np.full(N_CHANNELS, np.nan); m[good] = med; q[good] = iqr
    return m, q
