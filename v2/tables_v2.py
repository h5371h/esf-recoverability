"""Multi-backbone tables for the paper.
  python tables_v2.py <agg_out_dir> <results_dir>  -> writes table_multi_body.tex (Table 1: per axis/severity, 3 backbones, naive/canon
  mean over MLP seeds) and table_envelope_body.tex (Table 3: baseline spread, linear probe, stage attribution, envelope)."""
import sys, json, pandas as pd, numpy as np
A, R = sys.argv[1], sys.argv[2]
summ = json.load(open(f"{A}/summary_multibackbone.json")); BB = [b for b in ["eegpt","labram","biot"] if b in summ]
NAME = {"eegpt":"EEGPT","labram":"LaBraM","biot":"BIOT"}
T = {b: pd.read_csv(f"{A}/table_{b}.csv", dtype={"severity": str}).set_index(["axis","severity","arm"]) for b in BB}
import sys as _s; _s.path.insert(0, "../../audit/src"); _s.path.insert(0, "../audit/src")
from statistical_tests import delong_p_value
def norm_sev(v):
    try: return f"{float(v):g}"
    except ValueError: return str(v)
STARS = {}
for b in BB:
    try: P = pd.read_csv(f"{R}/per_recording_predictions_{b}_mlp_s7.csv", dtype={"severity": str})
    except FileNotFoundError: continue
    P["severity"] = P.severity.map(norm_sev); pv = {}
    for (ax, sev), g in P.groupby(["axis", "severity"]):
        if ax in ("broadband", "bandwidth"): continue
        c = g[g.arm=="canonicalized"].set_index("rec_id"); n = g[g.arm=="naive"].set_index("rec_id"); idx = c.index.intersection(n.index)
        if len(idx) == 0 or np.allclose(c.loc[idx].logit.values, n.loc[idx].logit.values): continue
        pv[(ax, sev)] = delong_p_value(c.loc[idx].label.values, c.loc[idx].logit.values, n.loc[idx].logit.values)[2]
    keys = list(pv); ps = np.array([pv[k] for k in keys]); m = len(ps); order = np.argsort(ps); adj = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]): prev = min(prev, ps[i] * m / (m - rank)); adj[i] = prev
    STARS[b] = {k: ("$^{***}$" if a < 0.001 else "$^{**}$" if a < 0.01 else "$^{*}$" if a < 0.05 else "") for k, a in zip(keys, adj)}
def cell(b, ax, sev, arm):
    try: r = T[b].loc[(ax, sev, arm)]; star = STARS.get(b, {}).get((ax, sev), "") if arm == "canonicalized" else ""; return f"{r.auroc_mean:.3f}{star}"
    except KeyError: return "--"
def row(label, ax, sev):
    return f"\\quad {label} & " + " & ".join(f"{cell(b,ax,sev,'naive')} & {cell(b,ax,sev,'canonicalized')}" for b in BB) + " \\\\"
L = [r"\midrule", r"\multicolumn{%d}{l}{\emph{(A1) Sampling rate}}\\" % (1+2*len(BB))]
L += [row(f"$f_n{{=}}{s}$", "sampling_rate", s) for s in ["128","200","256","512"]]
L += [r"\midrule", r"\multicolumn{%d}{l}{\emph{(A2) Per-channel gain}}\\" % (1+2*len(BB))]
L += [row(f"$g{{=}}{s}$", "calibration", s) for s in ["0.5","0.75","1.5","2"]]
L += [r"\midrule", r"\multicolumn{%d}{l}{\emph{(A3) Mains interference ($\mu$V)}}\\" % (1+2*len(BB))]
L += [row(f"$a{{=}}{int(float(s))}$", "power_line", s) for s in ["5","15","30","60"]]
L += [r"\midrule", r"\multicolumn{%d}{l}{\emph{(A4) Electrode montage}}\\" % (1+2*len(BB))]
L += [row(s, "montage", s) for s in ["legacy16","bipolar18"]]
L += [r"\midrule", r"\multicolumn{%d}{l}{\emph{(A5) Broadband noise (arms identical)}}\\" % (1+2*len(BB))]
L += [row(f"SNR ${int(float(s))}$\\,dB", "broadband", s) for s in ["-5","0","5","10","20"]]
open("table_multi_body.tex","w").write("\n".join(L)+"\n")
H = [r"\begin{tabular}{l" + "cc"*len(BB) + "}", r"\toprule",
     "Condition & " + " & ".join(r"\multicolumn{2}{c}{%s}" % NAME[b] for b in BB) + r" \\",
     " & " + " & ".join("Naive & Canon." for _ in BB) + r" \\"]
open("table_multi_header.tex","w").write("\n".join(H)+"\n")
open("table_multi_full.tex","w").write("\n".join(H + L + [r"\bottomrule", r"\end{tabular}"])+"\n")
# Table 3: baselines + attribution + envelope
E = []
for b in BB:
    m = summ[b]["mlp"]; lin = summ[b].get("linear",{}).get("baseline_mean", float("nan")); at = summ[b]["stage_attribution"]
    outside = summ[b]["envelope_A1_A4_outside"]
    def rec(ax): return f"{at[ax]['recovery']:+.3f}" if ax in at else "--"
    SHORT = {"sampling_rate": lambda v: f"$f_n{{=}}{v}$", "calibration": lambda v: f"$g{{=}}{v}$", "power_line": lambda v: f"$a{{=}}{v}$", "montage": lambda v: v}
    names = ", ".join(SHORT[k.split("/")[0]](k.split("/")[1]) for k in outside)
    E.append(f"{NAME[b]} & {m['baseline_mean']:.3f} ({m['baseline_min']:.3f}--{m['baseline_max']:.3f}) & {lin:.3f} & {rec('sampling_rate')} & {rec('calibration')} & {rec('power_line')} & {rec('montage')} & {len(outside)}{(': ' + names) if outside else ''} \\\\")
open("table_envelope_body.tex","w").write("\n".join(E)+"\n")
EH = [r"\begin{tabular}{lccccccl}", r"\toprule",
      r"Backbone & Baseline & Lin. & \multicolumn{4}{c}{Recovery at worst severity} & Outside \\",
      r"& & & $f_n$ & $g$ & mains & montage & \\", r"\midrule"]
open("table_envelope_full.tex","w").write("\n".join(EH + E + [r"\bottomrule", r"\end{tabular}"])+"\n")
print("\n".join(L[:6])); print("..."); print("\n".join(E))

# Supplement Table S1: per-backbone full metrics (AUROC / bal acc / ECE, naive vs canon, mean over MLP seeds)
def cell3(b, ax, sev, arm):
    try: r = T[b].loc[(ax, sev, arm)]; return f"{r.auroc_mean:.3f} & {r.bal_acc:.3f} & {r.ece:.3f}"
    except KeyError: return "-- & -- & --"
ROWS = [("A1", "Sampling rate $f_n$ (Hz)", "sampling_rate", ["128","200","256","512"]), ("A2", "Gain $g$", "calibration", ["0.5","0.75","1.5","2"]),
        ("A3", "Mains $a$ ($\\mu$V)", "power_line", ["5","15","30","60"]), ("A4", "Montage", "montage", ["legacy16","bipolar18"]),
        ("A5", "Broadband SNR (dB)", "broadband", ["-5","0","5","10","20"]), ("B", "Low-pass cutoff (Hz)", "bandwidth", ["15","20","25","30","40","50","64","80","100"])]
for b in BB:
    S1 = [r"\begin{tabular}{lcccccc}", r"\toprule", r"Condition & \multicolumn{3}{c}{Naive} & \multicolumn{3}{c}{Canonicalized} \\",
          r"& AUROC & bal.\ acc. & ECE & AUROC & bal.\ acc. & ECE \\"]
    for tag, label, ax, sevs in ROWS:
        S1 += [r"\midrule", r"\multicolumn{7}{l}{\emph{(%s) %s}}\\" % (tag, label)]
        for sv in sevs:
            S1.append(f"\\quad {sv} & {cell3(b, ax, sv, 'naive')} & {cell3(b, ax, sv, 'canonicalized')} \\\\")
    S1 += [r"\bottomrule", r"\end{tabular}"]
    open(f"table_S1_{b}.tex", "w").write("\n".join(S1) + "\n")
print("S1 tables:", BB)

# Table 1: delta AUROC per cell as a coloured spreadsheet (needs \usepackage[table]{xcolor} + colortbl in the paper)
DCELLS = [("sampling_rate", "A1 rate (Hz)", ["128","200","256","512"]), ("calibration", "A2 gain ($\\times$)", ["0.5","0.75","1.5","2"]),
          ("power_line", "A3 mains ($\\mu$V)", ["5","15","30","60"]), ("montage", "A4 montage", ["legacy16","bipolar18"]), ("broadband", "A5 SNR (dB)", ["-5","0","5","10","20"])]
def basev(b):
    r = T[b].loc[("sampling_rate", "250", "canonicalized")]; return float(r.auroc_mean)
def shade(d):  # 0 .. 100 percent of the loss colour, saturating at a loss of 0.5
    return int(min(100, round(max(0.0, -d) / 0.5 * 100)))
H = [r"\begin{tabular}{@{}ll" + "".join("c"*len(c[2]) + (r"@{\hspace{5pt}}" if i < len(DCELLS)-1 else "") for i, c in enumerate(DCELLS)) + "@{}}", r"\toprule",
     " & & " + " & ".join(r"\multicolumn{%d}{c}{%s}" % (len(c[2]), c[1]) for c in DCELLS) + r" \\",
     " & & " + " & ".join(" & ".join(sv.replace("legacy16","leg16").replace("bipolar18","bip18") for sv in c[2]) for c in DCELLS) + r" \\", r"\midrule"]
L = []
for b in BB:
    for arm in ["naive", "canonicalized"]:
        cells = []
        for ax, _, sevs in DCELLS:
            for sv in sevs:
                if ax == "broadband" and arm == "naive": cells.append(r"\cellcolor{gray!12}"); continue
                try: d = float(T[b].loc[(ax, sv, arm)].auroc_mean) - basev(b)
                except KeyError: cells.append("--"); continue
                txt = f"{d:+.2f}".replace("-", "$-$") if abs(d) >= 0.005 else "0"
                p = shade(d); col = r"\cellcolor{loss!%d}" % p; fg = r"\color{white}" if p > 55 else ""
                cells.append(col + fg + (r"\textbf{%s}" % txt if (arm == "canonicalized" and d < -0.01) else txt))
        name = (r"\textcolor{%s}{\textbf{%s}}" % ("col" + b, NAME[b])) if arm == "naive" else ""
        L.append(name + " & " + arm + " & " + " & ".join(cells) + r" \\")
    L.append(r"\addlinespace[3pt]")
open("table_delta_full.tex", "w").write("\n".join(H + L[:-1] + [r"\bottomrule", r"\end{tabular}"]) + "\n"); print("delta table written")
