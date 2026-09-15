"""Stage 3b (v2.1): score cached cell features with one or more probes.
Writes per_recording_predictions_<tag>.csv and sweep_<tag>.csv in the v1 schema."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esf_v2.backbone import pool_groups

def ece(p, y, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i+1])
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)
def boot_auc(y, s, n=1000, seed=7):
    rng = np.random.default_rng(seed); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; out = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); out.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def load_probe(path):
    ck = torch.load(path, weights_only=False)
    if ck.get("probe", "mlp") == "mlp":
        D = ck["head_state"]["0.weight"].shape[1]; head = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)); head.load_state_dict(ck["head_state"]); head.eval()
        f = lambda X: head(torch.tensor(X, dtype=torch.float32)).squeeze(1).detach().numpy()
    else:
        w, b = ck["coef"], ck["intercept"]; f = lambda X: X @ w + b
    return f, ck["scaler_mean"], ck["scaler_scale"], ck["platt_a"], ck["platt_b"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, default=Path("cells")); ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True); ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(); a.out.mkdir(exist_ok=True, parents=True)
    f, sm, ss, pa, pb = load_probe(a.head)
    labels = {r["rec_id"]: int(r["label"]) for r in csv.DictReader(open(a.cells / "labels.csv"))}
    rows = []
    for axd in sorted(a.cells.iterdir()):
        if not axd.is_dir(): continue
        for sevd in sorted(axd.iterdir()):
            for armd in sorted(sevd.iterdir()):
                for fp in sorted(armd.glob("*.npy")):
                    wf = np.load(fp).astype(np.float32); G = pool_groups(wf)
                    if G.shape[0] == 0: continue
                    Gs = (G - sm) / np.where(ss > 1e-9, ss, 1.0)
                    with torch.no_grad(): gl = f(Gs)
                    logit = float(np.mean(gl)); var = float(np.var(gl)) if len(gl) > 1 else 0.0
                    rows.append([fp.stem, axd.name, sevd.name, armd.name, labels[fp.stem], f"{logit:.6f}", f"{1/(1+np.exp(-(pa*logit+pb))):.6f}", f"{var:.6f}"])
    per = a.out / f"per_recording_predictions_{a.tag}.csv"
    with open(per, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rec_id", "axis", "severity", "arm", "label", "logit", "prob", "var_logit"]); w.writerows(rows)
    df = pd.read_csv(per, dtype={"severity": str}); out = []
    for (ax, sev, arm), g in df.groupby(["axis", "severity", "arm"], sort=False):
        y, s, p = g.label.values, g.logit.values, g.prob.values; lo, hi = boot_auc(y, s)
        out.append([ax, sev, arm, round(roc_auc_score(y, s), 6), round(balanced_accuracy_score(y, p > 0.5), 6), round(ece(p, y), 6), len(g), round(lo, 6), round(hi, 6)])
    pd.DataFrame(out, columns=["axis", "severity", "arm", "auroc", "bal_acc", "ece", "n_recordings", "ci_lo", "ci_hi"]).to_csv(a.out / f"sweep_{a.tag}.csv", index=False)
    print("scored", a.tag, len(rows), "rows", flush=True)

if __name__ == "__main__":
    main()
