"""
TUEV recording-level IED-presence label parser.

Single-source-of-truth for converting a TUEV `.edf` path into a binary
"IED present (1) / background only (0) / drop (None)" label.

Used by:
  * apps/training/vertex/train_head_c_eegpt.py — Head C v1 training
  * apps/training/spmb_acquisition_shift/eval_loop.py — SPMB IED-task sweep

TUEV v2.0.1 ships TWO recording layouts in the same release:

  (a) `eval/` split — recording-level label encoded in filename prefix:
        <label>_<patient>_<event_letter>_<seg>.edf
        label ∈ {spsw, gped, pled, bckg, eyem, artf, null}

  (b) `train/` split — filename is patient-keyed, label per-segment in
      the `.rec` companion:
        <patient>_<seg>.edf  +  <patient>_<seg>.rec
        .rec rows: "channel_idx,start_sec,stop_sec,label_code"
        per TUEV README:
          1=null  2=spsw  3=gped  4=pled  5=eyem  6=artf  7=bckg

Recording-level positive iff ANY segment carries spsw/gped/pled
(filename prefix in {spsw, gped, pled} or .rec code ∈ {2, 3, 4}).
Recording-level negative iff prefix == bckg or all .rec segments are
code 7 (and none are 2/3/4).
Anything else (eyem, artf, null) returns None — the recording is
non-classifiable for binary IED-presence.

Edge cases handled:
  * Filename without a recognised prefix AND no `.rec` companion → None
  * `.rec` with only drop codes (null/eyem/artf) → None
  * Missing / unreadable `.rec` → None (logged at WARNING)
  * Mixed `.rec` (bckg + IED segments) → 1 (IED wins; matches Head C v1
    training, see train_head_c_eegpt.py:_label_from_rec)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Filename-prefix vocabulary (eval/ split layout) ────────────────────────
IED_LABELS = frozenset({"spsw", "gped", "pled"})
NEG_LABELS = frozenset({"bckg"})
DROP_LABELS = frozenset({"eyem", "artf", "null"})

# ── `.rec` integer code vocabulary (train/ split layout) ───────────────────
TUEV_REC_IED_CODES = frozenset({2, 3, 4})    # spsw, gped, pled
TUEV_REC_BCKG_CODE = 7
TUEV_REC_DROP_CODES = frozenset({1, 5, 6})   # null, eyem, artf


def _label_from_rec(rec_path: Path) -> Optional[int]:
    """Parse a TUEV `.rec` file. See module docstring for label rule."""
    try:
        text = rec_path.read_text(errors="replace")
    except Exception as e:
        logger.warning(f"  rec read fail {rec_path.name}: {e}")
        return None
    saw_ied = False
    saw_bckg = False
    saw_anything = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            code = int(parts[-1])
        except ValueError:
            continue
        saw_anything = True
        if code in TUEV_REC_IED_CODES:
            saw_ied = True
            break  # one IED segment is enough to mark the recording positive
        if code == TUEV_REC_BCKG_CODE:
            saw_bckg = True
    if saw_ied:
        return 1
    if saw_bckg:
        return 0
    return None if not saw_anything else None  # only drop codes — skip


def label_from_path(p: Path) -> Optional[int]:
    """Recording-level binary IED label for a TUEV `.edf` file.

    Tries (a) filename prefix first (eval/ layout), then (b) `.rec`
    companion file (train/ layout). Returns None for non-classifiable
    recordings (drop labels, missing companion, no prefix match).
    """
    prefix = p.stem.split("_", 1)[0].lower() if "_" in p.stem else ""
    if prefix in IED_LABELS:
        return 1
    if prefix in NEG_LABELS:
        return 0
    if prefix in DROP_LABELS:
        return None
    rec = p.with_suffix(".rec")
    if rec.exists():
        return _label_from_rec(rec)
    return None


def patient_id(p: Path) -> str:
    """Stable per-recording patient ID. TUEV stores recordings under
    .../edf/{train,eval}/<patient>/<file>.edf — the parent directory
    name is the patient. Falls back to filename prefix when the layout
    differs (e.g. flat directories during exploratory runs)."""
    parent = p.parent.name
    if parent and parent != "edf":
        return parent
    stem = p.stem
    return stem.split("_", 1)[0] if "_" in stem else stem
