"""
ESF Amplitude Normalization
===========================
Robust z-score using per-channel median and IQR.
Device-invariant: preserves inter-channel relationships.

Architecture doc invariant:
  ALL downstream features use normalized values. NEVER raw µV.
"""

from __future__ import annotations
import numpy as np


def robust_zscore(
    signals: np.ndarray,
    exclude_edges_sec: float = 10.0,
    sfreq: float = 250.0,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Normalize each channel to robust z-score: (x - median) / IQR.

    Args:
        signals:           shape (n_ch, n_samples), raw µV
        exclude_edges_sec: seconds to exclude at start/end (movement artifacts)
        sfreq:             sampling rate in Hz

    Returns:
        normalized:  float32 array same shape as signals
        medians:     per-channel median values (for de-normalization)
        iqrs:        per-channel IQR values (for de-normalization)
    """
    n_ch, n_samp = signals.shape
    edge = int(exclude_edges_sec * sfreq)
    # Core signal: exclude first and last `edge` samples
    core_start = min(edge, n_samp // 4)
    core_end   = max(n_samp - edge, 3 * n_samp // 4)
    core = signals[:, core_start:core_end] if core_end > core_start else signals

    medians: list[float] = []
    iqrs:    list[float] = []

    for ch in range(n_ch):
        ch_core = core[ch]
        med = float(np.median(ch_core))
        q75, q25 = np.percentile(ch_core, [75, 25])
        iqr = float(q75 - q25)
        medians.append(med)
        iqrs.append(iqr if iqr > 0 else 1.0)  # guard against flat channels

    medians_arr = np.array(medians, dtype=np.float32)
    iqrs_arr    = np.array(iqrs,    dtype=np.float32)

    normalized = (signals - medians_arr[:, None]) / iqrs_arr[:, None]
    return normalized.astype(np.float32), medians, iqrs


def denormalize(
    normalized: np.ndarray,
    medians: list[float],
    iqrs: list[float],
) -> np.ndarray:
    """Reverse robust_zscore to recover original µV amplitudes."""
    medians_arr = np.array(medians, dtype=np.float32)
    iqrs_arr    = np.array(iqrs,    dtype=np.float32)
    return (normalized * iqrs_arr[:, None] + medians_arr[:, None]).astype(np.float32)
