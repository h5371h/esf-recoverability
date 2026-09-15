"""Evaluate a probe on the cached EVAL features (clean, canonicalized) -> AUROC. Used for the label-permutation negative control.
Usage: python eval_probe.py features_<bb> head.pt"""
import sys, csv, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent)); from esf_v2.backbone import pool_recording
F, H = Path(sys.argv[1]), sys.argv[2]; ck = torch.load(H, weights_only=False)
rows = list(csv.DictReader(open(F / "eval_index.csv"))); X, y = [], []
for r in rows:
    f = np.load(F / "eval" / f"{r['rec_id']}.npy").astype(np.float32)
    if f.shape[0]: X.append(pool_recording(f)); y.append(int(r["label"]))
X, y = np.stack(X), np.array(y); Xs = (X - ck["scaler_mean"]) / np.where(ck["scaler_scale"] > 1e-9, ck["scaler_scale"], 1.0)
if ck.get("probe", "mlp") == "mlp":
    D = ck["head_state"]["0.weight"].shape[1]; head = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)); head.load_state_dict(ck["head_state"]); head.eval()
    with torch.no_grad(): s = head(torch.tensor(Xs, dtype=torch.float32)).squeeze(1).numpy()
else: s = Xs @ ck["coef"] + ck["intercept"]
print(f"{H}: eval AUROC {roc_auc_score(y, s):.4f} (n={len(y)})")
