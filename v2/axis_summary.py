"""Compact per-backbone, per-axis summary for writing the Results text. Usage: python axis_summary.py agg/ results/"""
import sys, glob, re, numpy as np, pandas as pd
sys.path.insert(0, "../audit/src"); from statistical_tests import delong_p_value
A, R = sys.argv[1], sys.argv[2]
def ns(v):
    try: return f"{float(v):g}"
    except ValueError: return str(v)
for f in sorted(glob.glob(f"{A}/table_*.csv")):
    bb = re.search(r"table_(\w+)\.csv", f).group(1); T = pd.read_csv(f, dtype={"severity": str})
    base = T[(T.axis=="sampling_rate")&(T.severity=="250")&(T.arm=="canonicalized")].iloc[0]
    print(f"\n=== {bb.upper()}  baseline AUROC {base.auroc_mean:.3f} [{base.auroc_min:.3f},{base.auroc_max:.3f}] bal {base.bal_acc:.3f} ece {base.ece:.3f} lin {base.auroc_linear:.3f}")
    try:
        P = pd.read_csv(f"{R}/per_recording_predictions_{bb}_mlp_s7.csv", dtype={"severity": str}); P["severity"] = P.severity.map(ns)
    except FileNotFoundError: P = None
    for ax in ["sampling_rate","calibration","power_line","montage","broadband","bandwidth"]:
        g = T[T.axis==ax]
        for sev in sorted(g.severity.unique(), key=lambda v: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else 0):
            n = g[(g.severity==sev)&(g.arm=="naive")]; c = g[(g.severity==sev)&(g.arm=="canonicalized")]
            if len(c)==0: continue
            c = c.iloc[0]; nv = n.iloc[0] if len(n) else None
            pstr = ""
            if P is not None and nv is not None and ax not in ("broadband","bandwidth"):
                q = P[(P.axis==ax)&(P.severity==sev)]; cc = q[q.arm=="canonicalized"].set_index("rec_id"); nn = q[q.arm=="naive"].set_index("rec_id"); idx = cc.index.intersection(nn.index)
                if len(idx) and not np.allclose(cc.loc[idx].logit, nn.loc[idx].logit): pstr = f" p={delong_p_value(cc.loc[idx].label.values, cc.loc[idx].logit.values, nn.loc[idx].logit.values)[2]:.1e}"
            print(f"  {ax:14s} {sev:>9s}  naive {nv.auroc_mean:.3f}/{nv.bal_acc:.2f}/{nv.ece:.2f}" if nv is not None else f"  {ax:14s} {sev:>9s}  naive   --", f"  canon {c.auroc_mean:.3f}[{c.auroc_min:.3f},{c.auroc_max:.3f}]/{c.bal_acc:.2f}/{c.ece:.2f}  lin {c.auroc_linear:.3f}{pstr}")
