"""Stage 2: Head A v2 — trained on TUAB *train* only (patient-stratified 90/10 train/val inside train).
Same recipe as v1 (Linear(1536,256)-ReLU-Dropout(0.2)-Linear(256,1), AdamW 1e-3/1e-4, BCE pos_weight,
25 epochs, batch 256, best-val-AUC checkpoint, StandardScaler on pooled features), plus Platt on val.
Also stores FIXED per-channel robust-scale constants (median of medians, median of IQRs over train) for the
calibration-axis naive arm."""
from __future__ import annotations
import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esf_v2.backbone import pool_recording

def ece(p, y, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i+1])
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def platt(logits, y):
    def obj(ab):
        z = ab[0]*logits + ab[1]; p = 1/(1+np.exp(-z)); p = np.clip(p, 1e-7, 1-1e-7)
        return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
    r = minimize(obj, x0=[1.0, 0.0], method="L-BFGS-B"); return float(r.x[0]), float(r.x[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("features"))
    ap.add_argument("--tuab", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("head_a_v2.pt"))
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--fixed-scale-n", type=int, default=300); ap.add_argument("--probe", choices=["mlp","linear"], default="mlp"); ap.add_argument("--permute-labels", action="store_true", help="negative control: shuffle training labels within the train split")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.features / "train_index.csv")))
    pids = sorted({r["rec_id"].split("_")[0] for r in rows}); rng = random.Random(a.seed); rng.shuffle(pids)
    val_p = set(pids[: int(round(0.1 * len(pids)))])
    X = {"train": [], "val": []}; Y = {"train": [], "val": []}
    for r in rows:
        f = np.load(a.features / "train" / f"{r['rec_id']}.npy")
        if f.shape[0] == 0: continue
        s = "val" if r["rec_id"].split("_")[0] in val_p else "train"
        X[s].append(pool_recording(f)); Y[s].append(int(r["label"]))
    Xtr, ytr = np.stack(X["train"]), np.array(Y["train"]); Xva, yva = np.stack(X["val"]), np.array(Y["val"])
    if a.permute_labels:
        rng_p = np.random.default_rng(1000 + a.seed); ytr = rng_p.permutation(ytr); yva = rng_p.permutation(yva)
        print("NEGATIVE CONTROL: training AND validation labels permuted (checkpoint selection and Platt see only permuted labels)", flush=True)
    print(f"train {len(ytr)} (pos {ytr.sum()})  val {len(yva)} (pos {yva.sum()})  patients {len(pids)}", flush=True)
    sc = StandardScaler().fit(Xtr); Xtr_s = sc.transform(Xtr).astype(np.float32); Xva_s = sc.transform(Xva).astype(np.float32)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.probe == "linear":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(Xtr_s, ytr)
        lv = clf.decision_function(Xva_s); hist = []; best_state = None
        head = None; coef, intercept = clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
    else:
        head = nn.Sequential(nn.Linear(Xtr_s.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        pos_w = torch.tensor([float((ytr == 0).sum()) / max(1, (ytr == 1).sum())])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt, yt = torch.tensor(Xtr_s), torch.tensor(ytr, dtype=torch.float32); Xv = torch.tensor(Xva_s)
        best, best_state, hist = -1, None, []
        for ep in range(a.epochs):
            head.train(); perm = torch.randperm(len(yt)); tl = 0.0
            for i in range(0, len(yt), 256):
                b = perm[i:i+256]; opt.zero_grad(); l = loss_fn(head(Xt[b]).squeeze(1), yt[b]); l.backward(); opt.step(); tl += float(l) * len(b)
            head.eval()
            with torch.no_grad(): lv = head(Xv).squeeze(1).numpy()
            auc = roc_auc_score(yva, lv); hist.append({"epoch": ep+1, "train_loss": tl/len(yt), "val_auc": float(auc)})
            if auc > best: best, best_state = auc, {k: v.clone() for k, v in head.state_dict().items()}
            print(f"  ep {ep+1:2d} loss {tl/len(yt):.4f} val_auc {auc:.4f}", flush=True)
    if head is not None:
        head.load_state_dict(best_state); head.eval()
        with torch.no_grad(): lv = head(Xv).squeeze(1).numpy()
    pa, pb = platt(lv, yva); pv = 1/(1+np.exp(-(pa*lv+pb)))
    metrics = {"val_auc": float(roc_auc_score(yva, lv)), "val_bal_acc": float(balanced_accuracy_score(yva, pv > 0.5)),
               "val_ece_platt": ece(pv, yva), "val_ece_raw": ece(1/(1+np.exp(-lv)), yva)}
    print("best:", metrics, flush=True)
    # fixed per-channel scale constants for the calibration-axis naive arm
    import mne; mne.set_log_level("ERROR")
    from esf_v2 import pipeline as P
    sub = [r for r in rows if r["rec_id"].split("_")[0] not in val_p]; rng.shuffle(sub); meds, iqrs = [], []
    for r in sub[: a.fixed_scale_n]:
        raw = mne.io.read_raw_edf(r["path"], preload=True, verbose=False)
        m, q = P.robust_constants((raw.get_data()*1e6).astype(np.float32), list(raw.ch_names), float(raw.info["sfreq"]), 60.0)
        meds.append(m); iqrs.append(q)
    fixed_med = np.nanmedian(np.stack(meds), 0); fixed_iqr = np.nanmedian(np.stack(iqrs), 0)
    torch.save({"probe": a.probe, "head_state": (head.state_dict() if head is not None else None), "coef": (coef if head is None else None), "intercept": (intercept if head is None else None), "scaler_mean": sc.mean_, "scaler_scale": sc.scale_, "platt_a": pa, "platt_b": pb,
                "fixed_med": fixed_med, "fixed_iqr": fixed_iqr, "metrics": metrics, "history": hist,
                "n_train": int(len(ytr)), "n_val": int(len(yva)), "val_patients": sorted(val_p), "seed": a.seed,
                "recipe": "Linear(1536,256)-ReLU-Dropout(0.2)-Linear(256,1); AdamW lr1e-3 wd1e-4; BCE pos_weight; 25 ep; batch 256; best val AUC; Platt on val"}, a.out)
    json.dump({"metrics": metrics, "history": hist, "n_train": int(len(ytr)), "n_val": int(len(yva))}, open(str(a.out) + ".json", "w"), indent=1)
    print("saved", a.out, flush=True)

if __name__ == "__main__":
    main()
