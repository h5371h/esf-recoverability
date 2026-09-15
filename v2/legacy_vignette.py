"""Real legacy recorder vignette: Natus .e files (proprietary) -> native parse -> contract -> each certified backbone -> probe.
No labels: reports native rate/channels, canonicalization outcome, the position of each recorder relative to each backbone's
envelope (from the sweep), and the per-recording logit vs the TUAB-eval clean logit distribution (percentile, and a KS test of
the 7 logits against the eval distribution). Usage: python legacy_vignette.py --edir ~/legacy --out results/legacy.json"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from esf_v2 import pipeline as P
from esf_v2.backbones import Backbone
from esf_v2.backbone import pool_groups

def load_probe(path):
    ck = torch.load(path, weights_only=False)
    if ck.get("probe", "mlp") == "mlp":
        D = ck["head_state"]["0.weight"].shape[1]; head = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)); head.load_state_dict(ck["head_state"]); head.eval()
        f = lambda X: head(torch.tensor(X, dtype=torch.float32)).squeeze(1).detach().numpy()
    else: f = lambda X: X @ ck["coef"] + ck["intercept"]
    return f, ck["scaler_mean"], ck["scaler_scale"], ck["platt_a"], ck["platt_b"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--edir", type=Path, required=True); ap.add_argument("--out", type=Path, default=Path("results/legacy.json"))
    ap.add_argument("--heads", type=str, default="eegpt=head_eegpt_mlp_s7.pt,labram=head_labram_mlp_s7.pt,biot=head_biot_mlp_s7.pt")
    ap.add_argument("--evalpreds", type=str, default="eegpt=results/per_recording_predictions_eegpt_mlp_s7.csv,labram=results/per_recording_predictions_labram_mlp_s7.csv,biot=results/per_recording_predictions_biot_mlp_s7.csv")
    a = ap.parse_args()
    from natus_e_parser import NatusEParser
    files = sorted(a.edir.glob("*.e")); seen = {}; recs = []
    for f in files:
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        if h in seen: print("duplicate skipped", f.name, "==", seen[h]); continue
        seen[h] = f.name
        r = NatusEParser().parse(str(f)); sig = np.asarray(r["signal"], dtype=np.float32).T; labels = list(r["channels"]); fs = float(r["sfreq"])
        x = P.canonical(sig, labels, fs, 50.0)   # Indian mains
        good = int((np.abs(x).sum(1) > 0).sum())
        recs.append({"file": f.name, "sha256": h[:16], "fs_native": fs, "n_ch_native": len(labels), "minutes": sig.shape[1]/fs/60, "esf_channels_resolved": good, "x": x})
        print(f.name, "fs", fs, "ch", len(labels), "min", round(sig.shape[1]/fs/60, 1), "resolved", good, flush=True)
    out = {"n_files": len(files), "n_unique": len(recs), "recordings": [{k: v for k, v in r.items() if k != "x"} for r in recs], "backbones": {}}
    heads = dict(kv.split("=") for kv in a.heads.split(",")); evalp = dict(kv.split("=") for kv in a.evalpreds.split(","))
    for name in ["eegpt", "labram", "biot"]:
        if not Path(heads[name]).exists() or not Path(evalp[name]).exists(): print("skip", name); continue
        bb = Backbone(name, "cuda" if torch.cuda.is_available() else "cpu"); f, sm, ss, pa, pb = load_probe(heads[name])
        ev = pd.read_csv(evalp[name], dtype={"severity": str}); clean = ev[(ev.axis=="sampling_rate")&(ev.severity=="250")&(ev.arm=="canonicalized")]
        sub = ev[(ev.axis=="sampling_rate")&(ev.severity=="128")&(ev.arm=="canonicalized")]
        logits = []
        for r in recs:
            G = pool_groups(bb.window_features(r["x"])); Gs = (G - sm) / np.where(ss > 1e-9, ss, 1.0)
            with torch.no_grad(): gl = f(Gs)
            logits.append(float(np.mean(gl)))
        logits = np.array(logits); pct = [float((clean.logit.values < l).mean()) for l in logits]
        ks = stats.ks_2samp(logits, clean.logit.values)
        out["backbones"][name] = {"legacy_logits": logits.tolist(), "legacy_probs": (1/(1+np.exp(-(pa*logits+pb)))).tolist(), "percentile_in_clean_eval": pct,
                                 "ks_vs_clean_eval_p": float(ks.pvalue), "eval_clean_auroc_cell": None,
                                 "envelope_125Hz_certified": None}
        print(name, "legacy logits", np.round(logits, 2), "KS p", round(ks.pvalue, 3), flush=True)
    json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)

if __name__ == "__main__":
    main()
