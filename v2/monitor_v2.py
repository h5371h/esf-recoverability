"""Label-free monitor test: does the across-group logit variance (var_logit) identify the naive-arm catastrophic
failures, and what does abstaining on it buy? Usage: python monitor_v2.py per_recording.csv out_prefix"""
import sys, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, dtype={"severity": str})
base = df[(df.axis=="sampling_rate")&(df.severity=="250")&(df.arm=="canonicalized")].set_index("rec_id")
base_correct = ((base.prob >= 0.5).astype(int) == base.label)
rows=[]
for (ax, sev, arm), g in df.groupby(["axis","severity","arm"], sort=False):
    g = g.set_index("rec_id").reindex(base.index)
    wrong = ((g.prob >= 0.5).astype(int) != g.label); cat = base_correct & wrong & ((g.prob - 0.5).abs() >= 0.30)
    n_cat = int(cat.sum())
    auc_var = roc_auc_score(cat, g.var_logit) if 0 < n_cat < len(g) else np.nan
    auc_conf = roc_auc_score(cat, -(g.prob-0.5).abs()) if 0 < n_cat < len(g) else np.nan
    res = dict(axis=ax, severity=sev, arm=arm, n_catastrophic=n_cat, auc_var_vs_cat=auc_var, auc_conf_vs_cat=auc_conf, auroc_all=roc_auc_score(g.label, g.logit))
    for q in (0.8, 0.5):
        keep = g.var_logit <= np.quantile(g.var_logit, q)
        res[f"caught_abstain{int(round((1-q)*100))}"] = float((cat & ~keep).sum()/n_cat) if n_cat else np.nan
        res[f"auroc_kept{int(round(q*100))}"] = roc_auc_score(g.label[keep], g.logit[keep]) if g.label[keep].nunique()==2 else np.nan
    rows.append(res)
t = pd.DataFrame(rows); t.to_csv(out + ".csv", index=False)
m = t[(t.arm=="naive") & (t.n_catastrophic >= 10)]
pd.set_option("display.width", 220)
print(m[["axis","severity","n_catastrophic","auc_var_vs_cat","auc_conf_vs_cat","caught_abstain20","auroc_all","auroc_kept80","caught_abstain50","auroc_kept50"]].round(3).to_string(index=False))
print("\nmedians over naive cells with >=10 catastrophic: var-AUC", round(m.auc_var_vs_cat.median(),3), "| conf-AUC", round(m.auc_conf_vs_cat.median(),3), "| caught@20%", round(m.caught_abstain20.median(),3), "| caught@50%", round(m.caught_abstain50.median(),3))
