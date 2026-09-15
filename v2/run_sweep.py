"""Stage 3: the v2 acquisition-shift sweep on TUAB eval (276), all 5 axes x severities x {naive, canonicalized}.
Operators act on RAW microvolt EDF at native rate (upstream); canonicalized = full ESF; naive = ESF with the
axis's corrective stage disabled (esf_v2.acquisition.NAIVE_SWITCH). Output schema identical to v1:
  per_recording_predictions_v2.csv : rec_id,axis,severity,arm,label,logit,prob,var_logit
  sweep_v2.csv                     : axis,severity,arm,auroc,bal_acc,ece,n_recordings,ci_lo,ci_hi"""
from __future__ import annotations
import argparse, csv, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esf_v2 import pipeline as P
from esf_v2.acquisition import AXES, NAIVE_SWITCH
from esf_v2.backbone import load_backbone, window_features, pool_groups

_CACHE = {}
def _raw(path):
    if path not in _CACHE:
        import mne; mne.set_log_level("ERROR")
        r = mne.io.read_raw_edf(path, preload=True, verbose=False)
        _CACHE.clear(); _CACHE[path] = ((r.get_data()*1e6).astype(np.float32), list(r.ch_names), float(r.info["sfreq"]))
    return _CACHE[path]

def cell_job(path, axis, sev, arm, fixed):
    """CPU: raw -> A_theta -> ESF(arm) -> normalised (19, n)."""
    try:
        sig, labels, fs = _raw(path)
        spec = AXES[axis]; fn = spec["fn"]
        rng = np.random.default_rng(zlib.crc32(f"{Path(path).stem}|{axis}|{sev}".encode()))
        sig2, labels2, fs2, lf2 = fn(sig, labels, fs, 60.0, sev, rng=rng)
        kw = dict(NAIVE_SWITCH[axis]) if arm == "naive" else {}
        if kw.get("norm") == "fixed": kw["fixed_scale"] = fixed
        x = P.run(sig2, labels2, fs2, lf2, **kw)
        return (path, axis, sev, arm, x)
    except Exception as e:
        return (path, axis, sev, arm, repr(e))

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuab", type=Path, required=True); ap.add_argument("--head", type=Path, default=Path("head_a_v2.pt"))
    ap.add_argument("--out", type=Path, default=Path("results")); ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--axes", type=str, default=None); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); a.out.mkdir(exist_ok=True, parents=True)
    ck = torch.load(a.head, weights_only=False)
    head = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)); head.load_state_dict(ck["head_state"]); head.eval()
    sm, ss, pa, pb = ck["scaler_mean"], ck["scaler_scale"], ck["platt_a"], ck["platt_b"]; fixed = (ck["fixed_med"], ck["fixed_iqr"])
    model, info = load_backbone("cuda"); print("backbone:", info, flush=True)
    edfs = sorted(str(p) for p in (a.tuab / "eval").rglob("*.edf"))
    if a.limit: edfs = edfs[: a.limit]
    labels = {p: (1 if "/abnormal/" in p else 0) for p in edfs}
    axes = a.axes.split(",") if a.axes else list(AXES)
    cells = [(ax, sev, arm) for ax in axes for sev in AXES[ax]["severities"] for arm in ("naive", "canonicalized")]
    print(f"{len(edfs)} recordings x {len(cells)} cells", flush=True)
    per_path = a.out / "per_recording_predictions_v2.csv"; done = set()
    if per_path.exists():
        for r in csv.DictReader(open(per_path)): done.add((r["rec_id"], r["axis"], r["severity"], r["arm"]))
    fh = open(per_path, "a", newline=""); w = csv.writer(fh)
    if not done: w.writerow(["rec_id", "axis", "severity", "arm", "label", "logit", "prob", "var_logit"])
    t0 = time.time(); n_done = 0
    for ax, sev, arm in cells:
        sev_s = AXES[ax]["fmt"](sev)
        todo = [p for p in edfs if (Path(p).stem, ax, sev_s, arm) not in done]
        if not todo: continue
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(cell_job, p, ax, sev, arm, fixed) for p in todo]
            for f in as_completed(futs):
                path, _, _, _, x = f.result()
                if isinstance(x, str): print(f"  FAIL {ax}/{sev_s}/{arm} {Path(path).name}: {x}", flush=True); continue
                wf = window_features(model, x, "cuda"); G = pool_groups(wf)
                if G.shape[0] == 0: continue
                Gs = (G - sm) / np.where(ss > 1e-9, ss, 1.0)
                with torch.no_grad(): gl = head(torch.tensor(Gs, dtype=torch.float32)).squeeze(1).numpy()
                logit = float(gl.mean()); var = float(gl.var()) if len(gl) > 1 else 0.0
                prob = float(1/(1+np.exp(-(pa*logit+pb))))
                w.writerow([Path(path).stem, ax, sev_s, arm, labels[path], f"{logit:.6f}", f"{prob:.6f}", f"{var:.6f}"]); n_done += 1
        fh.flush(); print(f"cell {ax}/{sev_s}/{arm} done  ({n_done} rows, {time.time()-t0:.0f}s)", flush=True)
    fh.close()
    # aggregate
    import pandas as pd
    df = pd.read_csv(per_path, dtype={"severity": str}); rows = []
    for (ax, sev, arm), g in df.groupby(["axis", "severity", "arm"], sort=False):
        y, s, p = g.label.values, g.logit.values, g.prob.values
        lo, hi = boot_auc(y, s)
        rows.append([ax, sev, arm, round(roc_auc_score(y, s), 6), round(balanced_accuracy_score(y, p > 0.5), 6), round(ece(p, y), 6), len(g), round(lo, 6), round(hi, 6)])
    pd.DataFrame(rows, columns=["axis", "severity", "arm", "auroc", "bal_acc", "ece", "n_recordings", "ci_lo", "ci_hi"]).to_csv(a.out / "sweep_v2.csv", index=False)
    print(pd.read_csv(a.out / "sweep_v2.csv").to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
