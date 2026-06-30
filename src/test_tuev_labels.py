"""
Unit tests for the TUEV recording-level IED-presence label parser used by
the SPMB Head C (IED-task) perturbation sweep.

No real EDFs needed — the parser is path/text only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tuev_labels import (  # noqa: E402
    label_from_path,
    patient_id,
)


# ── eval/ split filename layout ────────────────────────────────────────────
@pytest.mark.parametrize("stem,expected", [
    ("spsw_00001234_a_001", 1),
    ("gped_00001234_a_001", 1),
    ("pled_00001234_a_001", 1),
    ("bckg_00001234_a_001", 0),
    ("eyem_00001234_a_001", None),  # drop
    ("artf_00001234_a_001", None),  # drop
    ("null_00001234_a_001", None),  # drop
    ("noprefix", None),             # no underscore → no prefix → no .rec → None
])
def test_eval_filename_prefix(tmp_path, stem, expected):
    """eval/ layout: label encoded in filename prefix; no .rec companion."""
    p = tmp_path / f"{stem}.edf"
    p.touch()
    assert label_from_path(p) == expected


# ── train/ split `.rec` companion layout ───────────────────────────────────
def test_rec_companion_only_bckg_is_negative(tmp_path):
    edf = tmp_path / "00001234_s001_t000.edf"
    edf.touch()
    rec = edf.with_suffix(".rec")
    rec.write_text("0,0.0,1.0,7\n1,1.0,2.0,7\n")
    assert label_from_path(edf) == 0


def test_rec_companion_any_ied_is_positive(tmp_path):
    edf = tmp_path / "00005555_s001_t000.edf"
    edf.touch()
    # bckg + spsw mix → recording is IED-positive (spsw wins).
    rec = edf.with_suffix(".rec")
    rec.write_text("0,0.0,1.0,7\n1,1.0,2.0,2\n")  # 2 = spsw
    assert label_from_path(edf) == 1


@pytest.mark.parametrize("code", [3, 4])  # gped, pled
def test_rec_companion_gped_pled_positive(tmp_path, code):
    edf = tmp_path / f"00007777_s001_t000_{code}.edf"
    edf.touch()
    rec = edf.with_suffix(".rec")
    rec.write_text(f"0,0.0,1.0,{code}\n")
    assert label_from_path(edf) == 1


@pytest.mark.parametrize("code", [1, 5, 6])  # null, eyem, artf
def test_rec_companion_only_drop_codes_is_none(tmp_path, code):
    edf = tmp_path / f"00008888_s001_t000_{code}.edf"
    edf.touch()
    rec = edf.with_suffix(".rec")
    rec.write_text(f"0,0.0,1.0,{code}\n")
    assert label_from_path(edf) is None


def test_rec_companion_missing_returns_none(tmp_path):
    """Filename has no recognised prefix and no .rec → cannot label."""
    edf = tmp_path / "00009999_s001_t000.edf"
    edf.touch()
    # No .rec companion next to it.
    assert label_from_path(edf) is None


def test_rec_companion_blank_and_comment_lines(tmp_path):
    edf = tmp_path / "00001111_s001_t000.edf"
    edf.touch()
    rec = edf.with_suffix(".rec")
    rec.write_text("# header comment\n\n0,0.0,1.0,7\n# trailing\n")
    assert label_from_path(edf) == 0


def test_rec_companion_malformed_lines_ignored(tmp_path):
    edf = tmp_path / "00002222_s001_t000.edf"
    edf.touch()
    rec = edf.with_suffix(".rec")
    rec.write_text(
        "garbage\n"
        "0,0.0\n"             # too few cols → skip
        "0,0.0,1.0,abc\n"     # non-int code → skip
        "0,0.0,1.0,2\n"       # valid spsw → positive
    )
    assert label_from_path(edf) == 1


# ── patient_id ─────────────────────────────────────────────────────────────
def test_patient_id_from_parent_dir(tmp_path):
    pdir = tmp_path / "00001234"
    pdir.mkdir()
    edf = pdir / "00001234_s001_t000.edf"
    edf.touch()
    assert patient_id(edf) == "00001234"


def test_patient_id_fallback_to_filename_prefix(tmp_path):
    # Parent dir is the literal "edf" → fall back to filename prefix.
    pdir = tmp_path / "edf"
    pdir.mkdir()
    edf = pdir / "spsw_00009999_a_001.edf"
    edf.touch()
    # First underscore-delimited token is "spsw" (the label, not the
    # patient) — this is the documented fallback; the per-recording sweep
    # eval doesn't care about cross-recording leakage since splits are
    # already fixed at training time and we evaluate every recording once.
    assert patient_id(edf) == "spsw"
