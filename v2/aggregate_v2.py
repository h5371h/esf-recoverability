"""Aggregate the multi-backbone, multi-probe sweeps into: (a) per-backbone Table rows = mean over the 3 MLP seeds with
[min,max] spread, linear probe alongside; (b) the deployment envelope per backbone (cells where canon AUROC is within
tol of that probe's baseline); (c) stage attribution = canon - naive per axis at the worst severity.
Usage: python aggregate_v2.py results/ out/"""
import sys, glob, json, re, numpy as np, pandas as pd
R, OUT = sys.argv[1], sys.argv[2]
import os; os.makedirs(OUT, exist_ok=True)
def norm_sev(v):
    try: return f"{float(v):g}"
    except ValueError: return str(v)
files = sorted(glob.glob(f"{R}/sweep_*_s*.csv"))
rows = []
for f in files:
    m = re.match(r".*sweep_(\w+?)_(mlp|linear)_s(\d+)\.csv", f)
    if m is None: continue
    bb, probe, seed = m.group(1), m.group(2), int(m.group(3)); bb = bb.replace("_bandwidth", "")  # bandwidth sweeps fold into their backbone
    d = pd.read_csv(f, dtype={"severity": str}); d["backbone"], d["probe"], d["seed"] = bb, probe, seed
    d["severity"] = d.severity.map(norm_sev); rows.append(d)
S = pd.concat(rows, ignore_index=True); S.to_csv(f"{OUT}/all_sweeps.csv", index=False)
def base_of(g): return float(g[(g.axis=="sampling_rate")&(g.severity=="250")&(g.arm=="canonicalized")].auroc.iloc[0])
summ = {}
for bb, gb in S.groupby("backbone"):
    summ[bb] = {}
    for probe, gp in gb.groupby("probe"):
        bases = {s: base_of(g) for s, g in gp.groupby("seed")}
        summ[bb][probe] = {"baseline_mean": float(np.mean(list(bases.values()))), "baseline_min": float(min(bases.values())), "baseline_max": float(max(bases.values())), "n_seeds": len(bases)}
    # mean table over MLP seeds
    mlp = gb[gb.probe=="mlp"]
    tab = mlp.groupby(["axis","severity","arm"]).agg(auroc_mean=("auroc","mean"), auroc_min=("auroc","min"), auroc_max=("auroc","max"), bal_acc=("bal_acc","mean"), ece=("ece","mean")).reset_index()
    lin = gb[gb.probe=="linear"][["axis","severity","arm","auroc"]].rename(columns={"auroc":"auroc_linear"})
    tab = tab.merge(lin, on=["axis","severity","arm"], how="left"); tab.to_csv(f"{OUT}/table_{bb}.csv", index=False)
    # envelope: canon cells within tol of baseline (per seed, then require all seeds)
    tol = 0.01; env = {}
    for s, g in mlp.groupby("seed"):
        b = base_of(g); c = g[g.arm=="canonicalized"]
        env[s] = {f"{r.axis}/{r.severity}": bool(b - r.auroc <= tol) for r in c.itertuples()}
    cells = sorted(set(k for e in env.values() for k in e))
    summ[bb]["envelope_tol_0.01"] = {k: all(env[s].get(k, False) for s in env) for k in cells}
    # harmonization envelope = the four correctable axes only; broadband and bandwidth are measurement axes
    summ[bb]["envelope_A1_A4_outside"] = sorted(k for k, v in summ[bb]["envelope_tol_0.01"].items() if not v and k.split("/")[0] in ("sampling_rate","calibration","power_line","montage"))
    # stage attribution: recovery (canon - naive) at the worst naive severity per axis (mean over seeds)
    attr = {}
    for ax, ga in mlp.groupby("axis"):
        piv = ga.pivot_table(index=["severity"], columns="arm", values="auroc", aggfunc="mean")
        if "naive" in piv and "canonicalized" in piv:
            w = piv.naive.idxmin(); attr[ax] = {"worst_severity": w, "naive": float(piv.naive[w]), "canon": float(piv.canonicalized[w]), "recovery": float(piv.canonicalized[w]-piv.naive[w])}
    summ[bb]["stage_attribution"] = attr
json.dump(summ, open(f"{OUT}/summary_multibackbone.json","w"), indent=1)
for bb in summ:
    print(f"== {bb}: baseline mlp {summ[bb].get('mlp',{}).get('baseline_mean',float('nan')):.3f} [{summ[bb].get('mlp',{}).get('baseline_min',float('nan')):.3f},{summ[bb].get('mlp',{}).get('baseline_max',float('nan')):.3f}] linear {summ[bb].get('linear',{}).get('baseline_mean',float('nan')):.3f}")
    for ax, a in summ[bb]["stage_attribution"].items(): print(f"   {ax:14s} worst {a['worst_severity']:>9s} naive {a['naive']:.3f} canon {a['canon']:.3f} recovery {a['recovery']:+.3f}")
    out = [k for k, v in summ[bb]["envelope_tol_0.01"].items() if not v]; print("   outside envelope (tol 0.01):", out or "none")
