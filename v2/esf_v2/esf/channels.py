"""
ESF Channel Set — 19-channel 10-20 system
==========================================
Architecture doc: 19 channels, 250 Hz, referential (common average).
Handles both old nomenclature (T3/T4/T5/T6) and new (T7/T8/P7/P8).
Internal storage uses old nomenclature for historical compatibility.
"""

from typing import Optional

# ── Canonical 19-channel set (10-20 system) ──────────────────────────────────
# Index is the ESF channel position — NEVER change order (breaks model compat).
STANDARD_CHANNELS: list[str] = [
    "Fp1",   # 0  — Frontal Polar Left
    "Fp2",   # 1  — Frontal Polar Right
    "F3",    # 2  — Frontal Left
    "F4",    # 3  — Frontal Right
    "C3",    # 4  — Central Left
    "C4",    # 5  — Central Right
    "P3",    # 6  — Parietal Left
    "P4",    # 7  — Parietal Right
    "O1",    # 8  — Occipital Left
    "O2",    # 9  — Occipital Right
    "F7",    # 10 — Anterior Temporal Left
    "F8",    # 11 — Anterior Temporal Right
    "T3",    # 12 — Mid Temporal Left   (= T7 in new nomenclature)
    "T4",    # 13 — Mid Temporal Right  (= T8)
    "T5",    # 14 — Posterior Temporal Left  (= P7)
    "T6",    # 15 — Posterior Temporal Right (= P8)
    "Fz",    # 16 — Frontal Midline
    "Cz",    # 17 — Central Midline
    "Pz",    # 18 — Parietal Midline
]

# Reverse lookup: canonical_name → index
CH_INDEX: dict[str, int] = {ch: i for i, ch in enumerate(STANDARD_CHANNELS)}

N_CHANNELS = len(STANDARD_CHANNELS)  # 19
assert N_CHANNELS == 19

# ── Channel aliases (old ↔ new nomenclature + common variants) ────────────────
# Maps any variant → canonical ESF name
CH_ALIASES: dict[str, str] = {
    # New → old (internal storage uses old)
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
    # Lowercase variants
    **{ch.lower(): ch for ch in STANDARD_CHANNELS},
    # With EEG prefix (some vendors prepend)
    **{f"EEG {ch}": ch for ch in STANDARD_CHANNELS},
    **{f"EEG {ch}-Ref": ch for ch in STANDARD_CHANNELS},
    **{f"EEG {ch}-LE": ch for ch in STANDARD_CHANNELS},
    **{f"EEG {ch}-Avg": ch for ch in STANDARD_CHANNELS},
    # New names with prefix
    "EEG T7": "T3", "EEG T7-Ref": "T3", "EEG T7-LE": "T3",
    "EEG T8": "T4", "EEG T8-Ref": "T4", "EEG T8-LE": "T4",
    "EEG P7": "T5", "EEG P7-Ref": "T5", "EEG P7-LE": "T5",
    "EEG P8": "T6", "EEG P8-Ref": "T6", "EEG P8-LE": "T6",
}


def resolve_channel_name(raw_label: str) -> Optional[str]:
    """
    Map a vendor channel label to its canonical ESF name.
    Returns None if the channel is not in the 10-20 set.

    Strategy:
    1. Exact match (case-sensitive)
    2. Alias lookup
    3. Case-insensitive match
    4. Strip prefixes/suffixes (EEG, -Ref, -LE, -A1+A2 linked-ear, -CAR, -M1, ...)
    5. Strip bipolar pair labels (e.g., "Fp1-F7" → Fp1 if first half matches)
    6. Strip vendor-specific quirks:
       - Indian vendors (RMS, Clarity, Allengers) often duplicate prefixes
         ("EEG-Fp1-A1" or "Ch1-Fp1" or "1.Fp1")
       - Chinese/Korean exports sometimes use "C3 - A1+A2" with spaces around dashes
       - Some Persyst exports include source-channel hints like "Fp1#1"
    """
    label = raw_label.strip()

    if label in CH_INDEX:
        return label
    if label in CH_ALIASES:
        return CH_ALIASES[label]

    label_lower = label.lower()
    for ch in STANDARD_CHANNELS:
        if ch.lower() == label_lower:
            return ch

    import re
    # Step A: strip leading channel-number prefixes ("1.Fp1", "Ch01-Fp1", "01:Fp1")
    s = re.sub(r"^\s*(ch|chan|channel)?\s*[0-9]+[\.\:\-_]\s*", "", label, flags=re.IGNORECASE)

    # Step B: strip leading modality prefix
    s = re.sub(r"^(EEG|EMG|ECG|EOG)\s*[-_:]?\s*", "", s, flags=re.IGNORECASE)

    # Step C: strip trailing reference qualifier
    #   variants: -Ref, -LE, -Avg, -A1, -A2, -A1+A2, -M1, -M2, -REF, -AVG, -CAR, -CLE,
    #             -REST, -ipsi, -contra, -CAR2, "#1" (Persyst stream index)
    s = re.sub(
        r"\s*[-_\s]*\(?\s*(Ref(erence)?|LE|Avg|Average|A1\+?A2|A1|A2|M1\+?M2|M1|M2|REF|AVG|CLE|CAR2?|REST|ipsi|contra)\s*\)?\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"#\d+$", "", s)  # Persyst stream-index suffix
    s = s.strip()

    if s in CH_INDEX:    return s
    if s in CH_ALIASES:  return CH_ALIASES[s]
    for ch in STANDARD_CHANNELS:
        if ch.lower() == s.lower():
            return ch

    # Step D: bipolar pair "Fp1-F7" — take the first half if it resolves
    if "-" in s and not s.startswith("-"):
        head = s.split("-", 1)[0].strip()
        if head in CH_INDEX:
            return head
        if head in CH_ALIASES:
            return CH_ALIASES[head]
        for ch in STANDARD_CHANNELS:
            if ch.lower() == head.lower():
                return ch

    # Step E: alphanumeric extraction — pulls "Fp1" out of "EEG-Fp1-Ref-(F3)" type messes
    m = re.search(
        r"\b(Fp[12]|F[3478z]|C[34z]|P[34z]|O[12]|T[3456]|T[78]|P[78])\b",
        s, flags=re.IGNORECASE,
    )
    if m:
        candidate = m.group(1)
        # Normalize case
        norm = candidate[0].upper() + candidate[1:].lower() if len(candidate) > 1 else candidate
        if norm in CH_INDEX:    return norm
        if norm in CH_ALIASES:  return CH_ALIASES[norm]

    return None
