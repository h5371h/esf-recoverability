"""Table 2 (failure-category prevalence) for several backbones x arms from the taxonomy profile CSVs.
Usage: python table_failure_multi.py <audit/data dir> <out.tex> eegpt=<suffix_or_empty> biot=_biot [labram=_labram]"""
import sys, pandas as pd
D, OUT = sys.argv[1], sys.argv[2]; specs = [a.split("=", 1) for a in sys.argv[3:]]
NAME = {"eegpt": "EEGPT", "labram": "LaBraM", "biot": "BIOT"}
CATS = [("robust_correct", "Robust-correct"), ("shift_recovered", "Shift-recovered"), ("calibration_collapse", "Calibration collapse"),
        ("axis_specific", "Axis-specific"), ("shift_induced_flip", "Shift-induced flip"), ("catastrophic_shift", "Catastrophic shift"), ("robust_wrong", "Robust-wrong")]
cols = []
for bb, suf in specs:
    for arm in ["naive", "canonicalized"]:
        p = pd.read_csv(f"{D}/per_recording_failure_profile{suf}_{arm}.csv"); cols.append((bb, arm, p))
L = [r"\begin{tabular}{l" + "cc" * len(specs) + "}", r"\toprule",
     "Category & " + " & ".join(r"\multicolumn{2}{c}{%s}" % NAME[b] for b, _ in specs) + r" \\",
     " & " + " & ".join("Naive & Canon." for _ in specs) + r" \\", r"\midrule"]
for key, label in CATS:
    cells = []
    for bb, arm, p in cols:
        n = int((p.category == key).sum()); nn = int(((p.category == key) & (p.label == 0)).sum()) if "label" in p else None
        cells.append(f"{n} ({nn}N)" if nn is not None else f"{n}")
    L.append(f"{label} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
