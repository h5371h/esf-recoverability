"""Class-conditional + marginal distances for the restated Theorem 1, per (axis, severity, arm).
Reads per_recording_predictions_v2.csv; writes pinsker_v2.csv + pinsker_v2_summary.json.
TV is computed two ways: Gaussian approximation on logits (as in v1) and an empirical histogram estimate."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, dtype={"severity": str})
ref = df[(df.axis == "sampling_rate") & (df.severity == "250") & (df.arm == "canonicalized")].set_index("rec_id")
auc0 = roc_auc_score(ref.label, ref.logit)
def tv_gauss(a, b):
    m1, s1, m2, s2 = a.mean(), a.std() + 1e-9, b.mean(), b.std() + 1e-9
    x = np.linspace(min(m1 - 6*s1, m2 - 6*s2), max(m1 + 6*s1, m2 + 6*s2), 4001)
    return 0.5 * np.trapezoid(np.abs(norm.pdf(x, m1, s1) - norm.pdf(x, m2, s2)), x)
def tv_hist(a, b, bins=30):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max()); e = np.linspace(lo, hi, bins + 1)
    pa, pb = np.histogram(a, e)[0] / len(a), np.histogram(b, e)[0] / len(b); return 0.5 * np.abs(pa - pb).sum()
def kl_gauss(a, b):  # KL(b || a)
    m1, s1, m2, s2 = a.mean(), a.std() + 1e-9, b.mean(), b.std() + 1e-9
    return float(np.log(s1 / s2) + (s2**2 + (m2 - m1)**2) / (2 * s1**2) - 0.5)
rows = []
for (ax, sev, arm), g in df.groupby(["axis", "severity", "arm"], sort=False):
    g = g.set_index("rec_id").reindex(ref.index).dropna()
    identity = (ax == "sampling_rate" and sev == "250") or (ax == "calibration" and sev == "1.0" and arm == "canonicalized") \
               or (ax == "power_line" and sev == "0.0" and arm == "canonicalized") or (ax == "montage" and sev == "full19")
    a0, a1 = ref.loc[g.index], g
    d = abs(roc_auc_score(a1.label, a1.logit) - auc0)
    pos, neg = a0.label == 1, a0.label == 0
    tv_m = tv_gauss(a0.logit.values, a1.logit.values); tv_p = tv_gauss(a0.logit[pos].values, a1.logit[pos].values); tv_n = tv_gauss(a0.logit[neg].values, a1.logit[neg].values)
    rows.append(dict(axis=ax, severity=sev, arm=arm, identity=identity, d_auroc=d,
                     tv_marginal=tv_m, tv_marginal_hist=tv_hist(a0.logit.values, a1.logit.values),
                     tv_pos=tv_p, tv_neg=tv_n, cc_bound=tv_p + tv_n,
                     pinsker_cc=np.sqrt(0.5*max(kl_gauss(a0.logit[pos].values, a1.logit[pos].values),0)) + np.sqrt(0.5*max(kl_gauss(a0.logit[neg].values, a1.logit[neg].values),0)),
                     implied_C=d / tv_m if tv_m > 1e-9 else np.nan, cc_holds=bool(d <= tv_p + tv_n + 1e-9)))
t = pd.DataFrame(rows); t.to_csv(out + ".csv", index=False)
m = t[~t.identity]
summ = dict(baseline_auroc=float(auc0), n_arms=int(len(m)), cc_bound_holds=int(m.cc_holds.sum()),
            implied_C_median=float(m.implied_C.median()), implied_C_max=float(m.implied_C.max()),
            worst_cc_slack=float((m.cc_bound - m.d_auroc).min()))
json.dump(summ, open(out + "_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1)); print(m.sort_values("d_auroc", ascending=False).head(12).to_string(index=False))
