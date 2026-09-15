"""
ESF Converter
=============
Converts a RawEEG (vendor adapter output) → ESFRecording.

Pipeline:
  1. Map vendor channels → 10-20 positions
  2. Resample to 250 Hz (if needed)
  3. Apply 50/60 Hz notch filter
  4. Common average reference
  5. Robust z-score normalization
  6. Assess channel quality
  7. Build ESFMetadata
"""

from __future__ import annotations

import os
import uuid
import logging
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from .schema import ESFRecording, ESFMetadata, RawEEG, ChannelQuality
from .channels import (
    STANDARD_CHANNELS, N_CHANNELS, CH_INDEX, resolve_channel_name
)
from .normalize import robust_zscore

logger = logging.getLogger(__name__)

ESF_SRATE       = 250.0
ESF_VERSION     = "1.0.0"
CHUNK_SAMPLES   = 2500      # 10 s × 250 Hz — matches Zarr chunk layout


def _resolve_converter_git_sha() -> Optional[str]:
    """
    Return the git SHA of the converter code, baked at build time via
    ESF_CONVERTER_GIT_SHA env var. Falls back to GIT_SHA for convenience.
    Returns None when unset (dev/local runs).
    """
    sha = (
        os.environ.get("ESF_CONVERTER_GIT_SHA")
        or os.environ.get("GIT_SHA")
        or ""
    ).strip()
    return sha or None


def detect_montage_type(channel_labels: list[str]) -> str:
    """
    Infer the input EDF's reference scheme from channel label patterns.

    Returns one of:
      'bipolar'                 — labels like "Fp1-F7", "T3-T5"
      'referential_avg'         — labels like "Fp1-Avg", "Fp1-CAR"
      'referential_linked'      — labels like "Fp1-A1", "Fp1-M1", "Fp1-A2"
      'referential_common'      — labels like "EEG FP1-REF", "EEG FP1-LE" (TUH/Natus)
      'referential'             — bare 10-20 labels ("Fp1", "F3", "Cz")
      'unknown'                 — none of the above heuristics matched
    """
    import re

    bipolar_re  = re.compile(r'^[A-Za-z]{1,4}\d*-[A-Za-z]{1,4}\d+$')
    avg_re      = re.compile(r'-(Avg|CAR|Ref|REF|avg|car)$', re.I)
    linked_re   = re.compile(r'-(A1|A2|M1|M2|Mastoid|LM|RM)$', re.I)
    common_re   = re.compile(r'^EEG\s+\w+-(?:REF|LE|AR)$', re.I)

    n = len(channel_labels)
    if n == 0:
        return "unknown"

    bipolar_count  = 0
    avg_count      = 0
    linked_count   = 0
    common_count   = 0
    bare_count     = 0

    from .channels import resolve_channel_name

    for lbl in channel_labels:
        s = lbl.strip()
        if common_re.match(s):
            common_count += 1
        elif avg_re.search(s):
            avg_count += 1
        elif linked_re.search(s):
            linked_count += 1
        elif bipolar_re.match(s) and resolve_channel_name(s) is None:
            # Looks like "X-Y" but doesn't resolve to a single canonical id → bipolar pair
            bipolar_count += 1
        elif resolve_channel_name(s) is not None:
            bare_count += 1

    majority = max(bipolar_count, avg_count, linked_count, common_count, bare_count)
    if majority == 0:
        return "unknown"
    if bipolar_count  == majority:
        return "bipolar"
    if avg_count      == majority:
        return "referential_avg"
    if linked_count   == majority:
        return "referential_linked"
    if common_count   == majority:
        return "referential_common"
    return "referential"


def _resample(data: np.ndarray, orig_rate: float, target_rate: float) -> np.ndarray:
    """Resample channel data (n_ch, n_samp) to target_rate.

    TUH corpora (notably TUSZ) ship at 256 Hz; ESF canonical is 250 Hz. For that
    exact step we use ``scipy.signal.resample_poly`` (polyphase rational
    resampling: up=125, down=128) instead of FFT ``resample``, which rings on
    long clinical clips and does not preserve the integer ratio cleanly.

    For other rates we keep FFT resampling as a general fallback.
    """
    if abs(orig_rate - target_rate) < 0.5:
        return data

    # Exact rational 256 Hz → 250 Hz (TUSZ / common TUH continuous EEG)
    if abs(orig_rate - 256.0) < 0.5 and abs(target_rate - ESF_SRATE) < 0.5:
        return sp_signal.resample_poly(data, up=125, down=128, axis=1).astype(np.float32)

    ratio = target_rate / orig_rate
    n_out = max(1, int(round(data.shape[1] * ratio)))
    resampled = sp_signal.resample(data, n_out, axis=1)
    return resampled.astype(np.float32)


def _notch_filter(
    data: np.ndarray,
    sfreq: float,
    line_freq: float = 50.0,
    q: float = 30.0,
) -> np.ndarray:
    """Apply notch filter at line_freq (and harmonics up to Nyquist)."""
    nyq = sfreq / 2.0
    out = data.copy()
    freq = line_freq
    while freq < nyq - 1:
        b, a = sp_signal.iirnotch(freq / nyq, q)
        out = sp_signal.filtfilt(b, a, out, axis=1)
        freq += line_freq
    return out.astype(np.float32)


def _common_average_ref(data: np.ndarray, good_mask: np.ndarray) -> np.ndarray:
    """Subtract mean of good channels from all channels."""
    good = data[good_mask]
    if good.shape[0] == 0:
        return data
    avg = good.mean(axis=0, keepdims=True)
    return (data - avg).astype(np.float32)


def _assess_amplitude_validity(
    prenorm_uv: np.ndarray,           # (n_ch, n_samp) µV, post-CAR pre-z
    low_uv: float = 0.5,
    high_uv: float = 200.0,
) -> dict:
    """
    Sanity-check the post-CAR signal magnitude against clinical EEG plausibility.

    Clinical scalp EEG amplitudes are typically 5-100 µV. We use [0.5, 200] as
    a wide safety net. Anything outside means a vendor / parser problem we
    cannot ignore — the canonical case is an EDF with blank physical_dimension
    field (MNE reads as V, our *1e6 inflates to ~1e8 µV).

    Returns dict consumed by ESFMetadata.amplitude_validity. The verdict
    determines whether downstream absolute-µV biomarkers will run.
    """
    valid = prenorm_uv[~np.all(np.isnan(prenorm_uv), axis=1)]
    if valid.size == 0:
        return {
            "median_abs_uv": None,
            "verdict": "unknown",
            "expected_range_uv": [low_uv, high_uv],
            "note": "no usable channels — cannot assess amplitude",
        }
    median_abs = float(np.nanmedian(np.abs(valid)))
    if median_abs < low_uv:
        verdict = "implausible_low"
        note = (f"median |signal| = {median_abs:.2e} µV is below {low_uv} µV — "
                "likely scale/unit problem (file may store data in mV/V or have "
                "an under-scaled gain). Absolute-µV biomarkers refused.")
    elif median_abs > high_uv:
        verdict = "implausible_high"
        note = (f"median |signal| = {median_abs:.2e} µV is above {high_uv} µV — "
                "likely scale/unit problem (file may have blank EDF "
                "physical_dimension causing MNE to read as V; *1e6 inflates "
                "by 1e6×). Absolute-µV biomarkers refused.")
    else:
        verdict = "ok"
        note = f"median |signal| = {median_abs:.2f} µV is within clinical range"
    return {
        "median_abs_uv": round(median_abs, 4),
        "verdict": verdict,
        "expected_range_uv": [low_uv, high_uv],
        "note": note,
    }


def _assess_quality(
    data: np.ndarray,       # (n_ch, n_samp) normalized
    sfreq: float = ESF_SRATE,
    bad_zscore_thresh: float = 5.0,   # channel is BAD if > X% of samples exceed this
    bad_frac: float = 0.30,
    noisy_frac: float = 0.10,
) -> list[int]:
    """
    Heuristic channel quality assessment post-normalization.
    Returns list of ChannelQuality ints, len=N_CHANNELS.
    """
    quality = []
    for ch in range(data.shape[0]):
        row = data[ch]
        if np.all(np.isnan(row)):
            quality.append(int(ChannelQuality.MISSING))
            continue
        frac_extreme = float(np.mean(np.abs(row[~np.isnan(row)]) > bad_zscore_thresh))
        if frac_extreme > bad_frac:
            quality.append(int(ChannelQuality.BAD))
        elif frac_extreme > noisy_frac:
            quality.append(int(ChannelQuality.NOISY))
        else:
            quality.append(int(ChannelQuality.GOOD))
    return quality


def to_esf(
    raw: RawEEG,
    recording_id: Optional[str] = None,
    apply_notch: bool = True,
    apply_car: bool = True,
) -> ESFRecording:
    """
    Convert a RawEEG to ESFRecording.

    Steps:
      1. Map vendor channels to 10-20 (missing → NaN column)
      2. Resample to 250 Hz
      3. Notch filter (50/60 Hz)
      4. Common average reference (good channels only)
      5. Robust z-score normalization
      6. Assess channel quality
      7. Build metadata

    Args:
        raw:          vendor adapter output
        recording_id: UUID string; auto-generated if None
        apply_notch:  apply line noise notch filter
        apply_car:    apply common average reference

    Returns:
        ESFRecording with 19 channels at 250 Hz, float32, normalized.
    """
    rid = recording_id or str(uuid.uuid4())

    # ── 0. Detect input montage type ─────────────────────────────────────────
    montage_type = detect_montage_type(list(raw.channel_labels))
    if montage_type == "bipolar":
        logger.warning(
            f"[{rid}] Bipolar montage detected — channel mapping will fail for most "
            "channels. The ESF canonical view will have most channels as NaN. "
            "Consider re-exporting as referential before upload."
        )

    # ── 1. Map vendor channels to ESF positions ───────────────────────────────
    esf_raw = np.full((N_CHANNELS, raw.signals.shape[1]), np.nan, dtype=np.float32)
    mapped: set[int] = set()
    unmapped_native: list[str] = []
    # Track unmapped channel vendor indices so we can preserve their raw
    # signals as companion channels (EKG/EOG/EMG/sphenoidal/etc.) instead
    # of silently dropping the data at the canonicalization boundary.
    unmapped_vendor_indices: dict[str, int] = {}

    for vendor_idx, label in enumerate(raw.channel_labels):
        canonical = resolve_channel_name(label)
        if canonical is None:
            logger.debug(f"Channel '{label}' not in 10-20 set — preserved as unmapped")
            unmapped_native.append(label)
            unmapped_vendor_indices[label] = vendor_idx
            continue
        esf_idx = CH_INDEX[canonical]
        if esf_idx in mapped:
            logger.warning(f"Duplicate mapping to {canonical} from '{label}' — skipped")
            unmapped_native.append(label)
            unmapped_vendor_indices[label] = vendor_idx
            continue
        esf_raw[esf_idx] = raw.signals[vendor_idx]
        mapped.add(esf_idx)

    n_mapped = len(mapped)
    logger.info(f"[{rid}] Mapped {n_mapped}/{N_CHANNELS} channels")
    if n_mapped < 10:
        logger.warning(f"[{rid}] Only {n_mapped} channels mapped — recording may be unusable")

    # ── 2. Resample to 250 Hz ─────────────────────────────────────────────────
    if abs(raw.sampling_rate - ESF_SRATE) > 0.5:
        # Resample only mapped (non-NaN) channels; keep NaN channels as-is
        esf_resampled = np.full(
            (N_CHANNELS, max(1, int(round(esf_raw.shape[1] * ESF_SRATE / raw.sampling_rate)))),
            np.nan, dtype=np.float32,
        )
        good_idx = [i for i in range(N_CHANNELS) if not np.all(np.isnan(esf_raw[i]))]
        if good_idx:
            resampled_good = _resample(esf_raw[good_idx], raw.sampling_rate, ESF_SRATE)
            for j, gi in enumerate(good_idx):
                n = min(resampled_good.shape[1], esf_resampled.shape[1])
                esf_resampled[gi, :n] = resampled_good[j, :n]
        esf_raw = esf_resampled
        logger.info(f"[{rid}] Resampled {raw.sampling_rate}→{ESF_SRATE}Hz: {esf_raw.shape}")

    # ── 3. Notch filter ───────────────────────────────────────────────────────
    if apply_notch:
        good_mask = ~np.all(np.isnan(esf_raw), axis=1)
        if good_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                esf_raw[good_mask] = _notch_filter(
                    esf_raw[good_mask], ESF_SRATE, float(raw.line_frequency)
                )

    # ── 3b. High-pass 0.5 Hz to remove DC drift before CAR ───────────────────
    # Without this, recordings with heavy DC drift (e.g., long polysomnography
    # captures, recordings with skin-potential artefact) leave CAR with a
    # shared low-frequency component that gets subtracted unevenly across
    # channels — decorrelating physically adjacent ones. Empirical effect:
    # TUH adj_corr 0.26 → 0.32 (+23%); Natus pathological recordings barely
    # move (their adj_corr collapse is a data-quality issue, not pipeline).
    # Strictly improves CAR conditioning, no clinical-band signal removed.
    if apply_car:
        good_mask = ~np.all(np.isnan(esf_raw), axis=1)
        if good_mask.any():
            hp_sos = sp_signal.butter(4, 0.5, btype="highpass", fs=ESF_SRATE, output="sos")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                esf_raw[good_mask] = sp_signal.sosfiltfilt(
                    hp_sos, esf_raw[good_mask], axis=1
                ).astype(np.float32)

    # ── 4. Common average reference ───────────────────────────────────────────
    if apply_car:
        good_mask = ~np.all(np.isnan(esf_raw), axis=1)
        if good_mask.sum() >= 2:
            esf_raw = _common_average_ref(esf_raw, good_mask)

    # ── 4b. Snapshot pre-normalization signals (µV) ───────────────────────────
    # Used by raw amplitude feature extraction (cplane derive step).
    # Captured after resampling, notch, HP, and CAR — same physical domain as
    # the normalized output but with absolute amplitude intact.
    esf_prenorm = esf_raw.copy()   # (19, n_samp) float32, µV, NaN for missing ch

    # ── 4c. Amplitude validity check ─────────────────────────────────────────
    # Defense against vendor EDFs with blank physical_dimension field where
    # MNE reads data as V (not µV) and the upstream *1e6 conversion inflates
    # by 1e6×. Clinical EEG is 5-100 µV typically; we use [0.5, 200] as a
    # wide safety net. Anything outside is flagged so downstream absolute-µV
    # biomarkers refuse to compute on it.
    amplitude_validity = _assess_amplitude_validity(esf_prenorm)

    # ── 5. Robust z-score normalization ──────────────────────────────────────
    # Operate on non-NaN channels only; NaN channels stay NaN
    good_mask = ~np.all(np.isnan(esf_raw), axis=1)
    medians = [0.0] * N_CHANNELS
    iqrs    = [1.0] * N_CHANNELS

    if good_mask.any():
        esf_norm_good, meds_good, iqrs_good = robust_zscore(
            esf_raw[good_mask], sfreq=ESF_SRATE
        )
        esf_raw[good_mask] = esf_norm_good
        gi = [i for i, g in enumerate(good_mask) if g]
        for j, idx in enumerate(gi):
            medians[idx] = meds_good[j]
            iqrs[idx]    = iqrs_good[j]

    # ── 6. Channel quality assessment ─────────────────────────────────────────
    quality = _assess_quality(esf_raw, sfreq=ESF_SRATE)
    # Force MISSING for channels that were never mapped
    for i in range(N_CHANNELS):
        if i not in mapped:
            quality[i] = int(ChannelQuality.MISSING)
            esf_raw[i] = np.nan

    usable_count = sum(
        1 for q in quality
        if q in (int(ChannelQuality.GOOD), int(ChannelQuality.NOISY))
    )

    # ── 7. Build metadata ─────────────────────────────────────────────────────
    duration_sec = esf_raw.shape[1] / ESF_SRATE

    # Identify companion (non-EEG) channels from the original label set that
    # didn't map to 10-20. Heuristic: common prefixes for ECG/EMG/EOG/RESP/SPO2
    # etc. Anything else stays in unmapped_native_channels verbatim.
    #
    # Substring rule (broadened 2026-06-17) covers vendor variations like
    # "EEG EKG1-REF" (TUH), "cardiac" (Indian vendors), "EKG-Le" (Natus).
    # The companion_writer + read_api / iplane detectors use the SAME
    # substrings + normalization; keep them in sync.
    companion_prefixes = (
        "ecg", "ekg", "emg", "eog", "cardiac", "resp", "spo", "pulse", "temp",
        "airflow", "snore", "abd", "thor", "chin", "leg",
    )
    companion_available: list[str] = []
    truly_unmapped: list[str] = []
    companion_signals_uv: dict[str, np.ndarray] = {}
    for label in unmapped_native:
        lower = label.lower().strip()
        if any(lower.startswith(p) or p in lower for p in companion_prefixes):
            companion_available.append(label)
            # Capture the raw vendor signal for this companion channel —
            # NOT resampled to 250 Hz, NOT referenced, just the µV time series
            # as the vendor recorded it. Preserves clinical fidelity for
            # downstream biomarkers (EKG QRS, EOG blinks, EMG bursts).
            vendor_idx = unmapped_vendor_indices.get(label)
            if vendor_idx is not None and vendor_idx < raw.signals.shape[0]:
                companion_signals_uv[label] = raw.signals[vendor_idx].astype(np.float32)
        else:
            truly_unmapped.append(label)

    meta = ESFMetadata(
        esf_version=ESF_VERSION,
        recording_id=rid,
        source_format=raw.source_format,
        source_vendor=raw.source_vendor,
        patient_age=raw.patient_age,
        patient_sex=raw.patient_sex,
        recording_type=raw.recording_type,
        recording_duration_sec=round(duration_sec, 2),
        sampling_rate_original=float(raw.sampling_rate),
        sampling_rate_canonical=ESF_SRATE,
        line_frequency=raw.line_frequency,
        channel_quality=quality,
        normalization_medians=medians,
        normalization_iqrs=iqrs,
        conversion_timestamp=datetime.now(timezone.utc).isoformat(),
        converter_version=ESF_VERSION,
        usable_channel_count=usable_count,
        usable_data_percentage=round(100.0 * usable_count / N_CHANNELS, 1),
        channel_names=list(STANDARD_CHANNELS),
        notes=(raw.notes or "") + f" | input_montage:{montage_type} | mapped:{n_mapped}/{N_CHANNELS}",
        # v1.0 provenance
        source_file_sha256=raw.source_file_sha256,
        source_file_path=raw.source_file_path,
        converter_git_sha=_resolve_converter_git_sha(),
        # v1.0 annotations / events passthrough (seconds-based, sample-rate-agnostic)
        annotations=list(raw.annotations or []),
        invalid_segments=list(raw.invalid_segments or []),
        # v1.0 companion + unmapped channels
        companion_available=companion_available,
        companion_blob_path=None,      # set by writer if companion zarr is produced
        unmapped_native_channels=truly_unmapped,
        # v1.0.1 amplitude validity (defense against blank physical_dimension EDFs)
        amplitude_validity=amplitude_validity,
    )

    # ── 7b. v1.1: remap source-rate Events to ESF (250 Hz) timeline ───────────
    # Sample-accurate Event objects from the vendor adapter carry their
    # sample indices at the SOURCE rate (raw.sampling_rate). Remap to the
    # canonical 250 Hz so downstream readers (viewer overlays, biomarker
    # joins, training-label remap) consume one sample-rate convention.
    # Seconds shadows are recomputed at the target rate inside the remap.
    # Drift bound: 1/250 s = 4 ms — below all clinical event-length floors.
    if raw.events_v11:
        try:
            from .annotation_remap import remap_events
            from .event import Event
            source_events = [Event.from_dict(d) for d in raw.events_v11]
            remapped = remap_events(source_events, raw.sampling_rate, ESF_SRATE)
            # Drop events whose remapped timing falls outside the ESF
            # signal's sample range — guards against EDF+ TAL records
            # referencing past-the-end times (a real failure mode).
            n_esf_samples = esf_raw.shape[1]
            kept = [e for e in remapped if 0 <= e.start_sample <= n_esf_samples]
            meta.events = [e.to_dict() for e in kept]
            if len(kept) < len(remapped):
                logger.warning(
                    f"[{rid}] Dropped {len(remapped) - len(kept)} events outside "
                    f"ESF sample range [0, {n_esf_samples}]"
                )
        except Exception as e:
            logger.warning(f"[{rid}] v1.1 event remap failed; v1.0 annotations remain: {e}")

    # ── 7c. v1.1: vendor metadata + acquisition state passthrough ─────────────
    # Pass through whatever the loader captured. These are ALREADY at
    # recording start time (vendor metadata is rate-agnostic; acquisition
    # state was derived from already-seconds-based event labels).
    if raw.vendor_metadata_raw:
        meta.vendor_metadata_raw = dict(raw.vendor_metadata_raw)
    if raw.acquisition_state:
        meta.acquisition_state = dict(raw.acquisition_state)

    rec = ESFRecording(signals=esf_raw.astype(np.float32), metadata=meta)
    # Attach pre-normalization signals as a non-serialized attribute.
    # cplane reads this immediately after to_esf() to extract amplitude features.
    rec._prenorm_signals_uv = esf_prenorm.astype(np.float32)  # type: ignore[attr-defined]
    # Attach companion channel signals (EKG/EOG/EMG/etc.) at the ORIGINAL
    # vendor sample rate. zarr_io.write_esf_zarr_local / _blob will write
    # them to companion.zarr alongside signals.zarr when present, and the
    # blob path back-populates metadata.companion_blob_path at write time.
    rec._companion_signals_uv = companion_signals_uv  # type: ignore[attr-defined]
    rec._companion_sample_rate = float(raw.sampling_rate)  # type: ignore[attr-defined]
    return rec
