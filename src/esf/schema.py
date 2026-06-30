"""
ESF Core Schema — dataclasses for ESFRecording, ESFMetadata, RawEEG

Version 1.0.0 — frozen 2026-04-16.

Stability contract
------------------
The shape of ESFRecording.signals (19 channels, float32, 250 Hz),
the normalization scheme (robust z-score), and the 10-20 channel order
(STANDARD_CHANNELS) are part of the stable v1.0 contract. They do not
change within the v1.x series.

Metadata fields added in v1.0 (source provenance, annotations,
invalid_segments, companion channels, unmapped_native_channels) are
additive and backward-compatible: ESFMetadata.from_dict silently
tolerates v0.x files missing these fields.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import numpy as np

from .channels import STANDARD_CHANNELS, N_CHANNELS


class ChannelQuality(enum.IntEnum):
    """Per-channel quality flags. Stored as int8 in Zarr."""
    GOOD    = 0  # Clean and reliable — use for all analyses
    NOISY   = 1  # Intermittent artifacts, recoverable — use with caution
    BAD     = 2  # Unreliable throughout — exclude, do not interpolate
    MISSING = 3  # Not recorded — exclude, flag in report


@dataclass
class ESFMetadata:
    """
    Metadata envelope stored in Zarr .zattrs and in the canonical meta.json.
    All fields are SCORE-aligned where applicable.

    v1.0.0 schema — stable for the v1.x series.
    """
    # ── Identity & version ────────────────────────────────────────────────────
    esf_version: str = "1.0.0"
    recording_id: str = ""                        # UUID

    # ── Source description (claimed; may be wrong for legacy files) ──────────
    source_format: str = "unknown"                # natus_e | edf | tuh_edf | bdf
    source_vendor: str = "unknown"                # natus | nihon_kohden | unknown
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None             # M | F | unknown
    recording_type: str = "unknown"               # routine | icu | ltm | unknown

    # ── Signal description (deterministic; computed from file) ────────────────
    recording_duration_sec: float = 0.0
    sampling_rate_original: float = 0.0
    sampling_rate_canonical: float = 250.0        # Always 250 Hz in ESF
    line_frequency: int = 50                      # 50 Hz (India) or 60 Hz
    channel_quality: list[int] = field(
        default_factory=lambda: [ChannelQuality.GOOD] * N_CHANNELS
    )                                             # len=19, values 0-3
    normalization_medians: list[float] = field(
        default_factory=lambda: [0.0] * N_CHANNELS
    )
    normalization_iqrs: list[float] = field(
        default_factory=lambda: [1.0] * N_CHANNELS
    )
    usable_channel_count: int = 0
    usable_data_percentage: float = 100.0
    channel_names: list[str] = field(
        default_factory=lambda: list(STANDARD_CHANNELS)
    )

    # ── Conversion provenance ────────────────────────────────────────────────
    conversion_timestamp: str = ""                # ISO 8601
    converter_version: str = "1.0.0"              # matches esf_version by default
    notes: str = ""

    # ── NEW in v1.0: source file traceability ────────────────────────────────
    # Cryptographic link back to the raw file in eeg-raw/ so ESF can always be
    # re-derived from the original and audits can verify provenance.
    source_file_sha256: Optional[str] = None
    source_file_path: Optional[str] = None        # e.g. "eeg-raw/<study_id>.edf"
    converter_git_sha: Optional[str] = None       # git commit that produced this ESF

    # ── NEW in v1.0: annotations and events ──────────────────────────────────
    # Preserved from the source file (EDF annotations, patient events,
    # photic stim markers, hyperventilation periods, etc.).
    # Each entry:
    #   {"onset_sec": float, "duration_sec": float,
    #    "label": str, "source": str}   # source in {"edf","mne","manual",...}
    annotations: list[dict] = field(default_factory=list)

    # Ranges of the recording that should be excluded from analysis.
    # Each entry:
    #   {"start_sec": float, "end_sec": float,
    #    "reason": str, "affected_channels": list[int]}
    invalid_segments: list[dict] = field(default_factory=list)

    # ── NEW in v1.0: companion (non-EEG) channels ────────────────────────────
    # EKG/EMG/EOG etc. that exist in the source but don't belong in the
    # 19-channel ESF signal tensor. Stored as a sidecar zarr group.
    companion_available: list[str] = field(default_factory=list)
    companion_blob_path: Optional[str] = None     # e.g. "<study_id>/companion.zarr"

    # ── NEW in v1.0: unmapped source channels ────────────────────────────────
    # Channel labels present in the source file that did not map to the
    # 10-20 canonical set. Preserved (not silently discarded) so future
    # adapters can recover them from the raw file.
    unmapped_native_channels: list[str] = field(default_factory=list)

    # ── NEW in v1.0.1: amplitude validity ────────────────────────────────────
    # Sanity check on the post-CAR pre-normalization signal magnitude. Some
    # vendor EDFs ship with a blank physical_dimension field; MNE then reads
    # them as Volts instead of microvolts, and the upstream *1e6 conversion
    # inflates the signal by 1e6×. Downstream absolute-µV biomarkers
    # (sharp_transient slope thresholds, burst-suppression amplitude floors,
    # ripple amplitude minima) silently emit garbage on such files unless we
    # refuse. This field carries the verdict so the gate and the report can
    # see it. Format:
    #   {"median_abs_uv": float, "verdict": "ok"|"implausible_high"|
    #    "implausible_low"|"unknown", "expected_range_uv": [low, high],
    #    "note": str}
    # `verdict == "ok"` is the only state where absolute-µV biomarkers run.
    amplitude_validity: dict = field(default_factory=dict)

    # ── NEW in v1.1: structured events (taxonomy + provenance) ───────────────
    # See libs/esf/event.py for the dataclass. Stored as list-of-dict so the
    # ESFMetadata dataclass stays a pure data container; `events` are reified
    # to Event via the events_typed() helper below. Coexists with the v1.0
    # `annotations` field — converters writing v1.1 SHOULD populate both for
    # one major-version window so downstream readers can migrate gradually.
    events: list[dict] = field(default_factory=list)

    # ── NEW in v1.1: vendor metadata round-trip preservation ─────────────────
    # Whatever the vendor file's native header contained that we couldn't
    # canonicalize. FieldTrip calls this `hdr.orig`; EEGLAB calls it
    # `EEG.etc.original_header`. We never lose information in conversion.
    # Schema is intentionally unconstrained — vendor formats are wild.
    vendor_metadata_raw: dict = field(default_factory=dict)

    # ── NEW in v1.1: acquisition state (clinically relevant epochs) ──────────
    # Shape:
    #   {
    #     "photic_stimulation": {
    #        "performed": bool,
    #        "epochs": [{"start_sec": float, "end_sec": float, "frequency_hz": float}],
    #     },
    #     "hyperventilation": {"performed": bool, "epochs": [...]},
    #     "eyes_state_log": [{"start_sec": float, "end_sec": float, "state": "open|closed"}],
    #     "sleep_stages": null | [{"start_sec", "end_sec", "stage": "W|N1|N2|N3|R"}]
    #   }
    # These are clinically load-bearing (a finding in the photic-stim
    # window is not the same as a finding outside it) and used to be
    # buried inside `annotations` as ad-hoc free-text labels.
    acquisition_state: dict = field(default_factory=dict)

    # ── NEW in v1.1: clinical context ────────────────────────────────────────
    # Shape:
    #   {
    #     "indication": str,              # routine_outpatient | seizure_evaluation | ...
    #     "recording_protocol": str,      # routine | extended | sleep_deprived | ambulatory
    #     "scheduled_duration_min": int,
    #     "clinical_question": str,       # free-text the referring physician supplied
    #   }
    # Distinct from `recording_type` (which is a high-level enum) — this
    # carries the per-study context that lets the report generator address
    # the question the EEG was ordered for.
    clinical_context: dict = field(default_factory=dict)

    # ── NEW in v1.1: per-channel rich info ───────────────────────────────────
    # Per-channel metadata richer than the parallel arrays in v1.0.
    # Shape (one dict per ESF channel, length 19, parallel to channel_names):
    #   {
    #     "esf_label": "Fp1",
    #     "original_label": "EEG Fp1-LE",
    #     "type": "eeg",   # eeg | eog | emg | ecg | trigger | reference
    #     "unit": "uV",
    #     "position_3d_mm": [x, y, z] | null,
    #     "reference": "common_average",
    #     "status": "good|fair|bad|missing",
    #     "impedance_ohms": float | null
    #   }
    # The existing parallel arrays (channel_names, channel_quality) stay
    # populated for v1.0 readers. Writers that emit v1.1 populate both.
    channels_typed: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (all fields, including v1.0 additions)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ESFMetadata":
        """Tolerant deserializer: ignores unknown keys, keeps defaults for missing ones."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ESFRecording:
    """
    The canonical in-memory EEG recording.

    signals: float32 array, shape (19, n_samples)
             Amplitude-normalized (robust z-score, median/IQR).
             19 channels in STANDARD_CHANNELS order.
             250 Hz. Referential (common average reference).
             NaN = missing channel (quality flag = MISSING).

    metadata: ESFMetadata — full provenance and normalization params.
    """
    signals: np.ndarray        # shape (19, n_samples), dtype float32
    metadata: ESFMetadata

    def __post_init__(self):
        if self.signals.ndim != 2:
            raise ValueError(f"signals must be 2D, got {self.signals.ndim}D")
        if self.signals.shape[0] != N_CHANNELS:
            raise ValueError(
                f"signals.shape[0] must be {N_CHANNELS} (ESF channels), "
                f"got {self.signals.shape[0]}"
            )
        if self.signals.dtype != np.float32:
            self.signals = self.signals.astype(np.float32)

    @property
    def n_samples(self) -> int:
        return self.signals.shape[1]

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.metadata.sampling_rate_canonical

    @property
    def good_channels(self) -> list[str]:
        """Channel names with quality GOOD or NOISY."""
        from .channels import STANDARD_CHANNELS
        return [
            STANDARD_CHANNELS[i]
            for i, q in enumerate(self.metadata.channel_quality)
            if q in (ChannelQuality.GOOD, ChannelQuality.NOISY)
        ]

    @property
    def usable_channel_indices(self) -> list[int]:
        return [
            i for i, q in enumerate(self.metadata.channel_quality)
            if q in (ChannelQuality.GOOD, ChannelQuality.NOISY)
        ]


@dataclass
class RawEEG:
    """
    Pre-conversion representation. Output of vendor adapters before ESF.
    Preserves original channel names, sample rate, and raw amplitudes (µV).

    v1.0 adds optional provenance and annotation passthrough so nothing
    from the source file is silently discarded during canonicalization.
    """
    signals: np.ndarray        # shape (n_original_channels, n_samples), float32, µV
    channel_labels: list[str]  # original vendor labels, len = n_original_channels
    sampling_rate: float       # Hz
    source_format: str         # natus_e | edf | bdf | tuh_edf
    source_vendor: str = "unknown"
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    recording_type: str = "unknown"
    line_frequency: int = 50
    original_filepath: str = ""
    notes: str = ""

    # ── NEW in v1.0: provenance passthrough ──────────────────────────────────
    source_file_sha256: Optional[str] = None
    source_file_path: Optional[str] = None        # blob path, e.g. "eeg-raw/<study>.edf"

    # ── NEW in v1.0: annotations and events from source file ────────────────
    # Each entry: {"onset_sec": float, "duration_sec": float, "label": str, "source": str}
    annotations: list[dict] = field(default_factory=list)

    # Each entry: {"start_sec": float, "end_sec": float, "reason": str, "affected_channels": list[str]}
    invalid_segments: list[dict] = field(default_factory=list)

    # Per-channel impedance (kΩ), keyed by original vendor label. Optional.
    impedances_kohm: Optional[dict] = None

    # ── NEW in v1.1: sample-accurate Event objects (carried into ESF) ────────
    # Each Event is stored at the source sample rate here; to_esf remaps
    # to the canonical 250 Hz on conversion. Coexists with `annotations`
    # (seconds-only v1.0 dicts) — both populated by the loader so v1.0
    # readers keep working unchanged.
    events_v11: list[dict] = field(default_factory=list)  # serialized Event dicts at source rate

    # ── NEW in v1.1: vendor-native header preservation ───────────────────────
    # FieldTrip hdr.orig pattern. Whatever the vendor format had that we
    # couldn't canonicalize. to_esf passes through to ESFMetadata.vendor_metadata_raw.
    vendor_metadata_raw: dict = field(default_factory=dict)

    # ── NEW in v1.1: acquisition state derived at load time ──────────────────
    # photic_stimulation / hyperventilation / eyes_state_log derived from
    # labelled events. to_esf passes through to ESFMetadata.acquisition_state.
    acquisition_state: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.signals.dtype != np.float32:
            self.signals = self.signals.astype(np.float32)
        if len(self.channel_labels) != self.signals.shape[0]:
            raise ValueError(
                f"channel_labels length {len(self.channel_labels)} != "
                f"signals.shape[0] {self.signals.shape[0]}"
            )
