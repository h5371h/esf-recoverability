import csv, sys, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sys.path.insert(0, "."); from esf_v2.backbone import pool_recording
bb = sys.argv[1]
def load(split):
    rows = list(csv.DictReader(open(f"features_{bb}/{split}_index.csv")))
    X = np.stack([pool_recording(np.load(f"features_{bb}/{split}/" + r["rec_id"] + ".npy").astype(np.float32)) for r in rows]); y = np.array([int(r["label"]) for r in rows]); return X, y
Xtr, ytr = load("train"); Xev, yev = load("eval")
sc = StandardScaler().fit(Xtr); Ztr, Zev = sc.transform(Xtr), sc.transform(Xev)
print(f"[{bb}] standardized-norm AUROC (eval): {roc_auc_score(yev, np.linalg.norm(Zev, axis=1)):.3f}")
pca = PCA(n_components=50, random_state=0).fit(Ztr); W = pca.transform(Zev) / np.sqrt(pca.explained_variance_)
print(f"[{bb}] PCA-50 whitened (Mahalanobis-like) norm AUROC (eval): {roc_auc_score(yev, np.linalg.norm(W, axis=1)):.3f}")
# distance to train mean vs to class-agnostic kNN density
from sklearn.neighbors import NearestNeighbors
nn_ = NearestNeighbors(n_neighbors=10).fit(Ztr); d, _ = nn_.kneighbors(Zev); print(f"[{bb}] 10-NN distance to train set AUROC (eval): {roc_auc_score(yev, d.mean(1)):.3f}")
aucs = []
for s in range(20):
    torch.manual_seed(s); h = nn.Sequential(nn.Linear(Ztr.shape[1], 256), nn.ReLU(), nn.Linear(256, 1)); h.eval()
    with torch.no_grad(): o = h(torch.tensor(Zev, dtype=torch.float32)).squeeze(1).numpy()
    aucs.append(roc_auc_score(yev, o))
aucs = np.array(aucs); print(f"[{bb}] untrained random MLP heads (20 inits): AUROC mean {aucs.mean():.3f} sd {aucs.std():.3f} min {aucs.min():.3f} max {aucs.max():.3f} | |auc-0.5| mean {np.abs(aucs-0.5).mean():.3f}")
