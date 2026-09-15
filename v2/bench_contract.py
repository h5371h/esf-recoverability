"""Deployment cost of the contract and of each backbone on CPU.
Times each ESF stage on real TUAB eval recordings (single process) and the per-recording inference cost of each
backbone on CPU (torch threads = 1 and = all). Usage: python bench_contract.py --tuab ~/tuab/edf --n 20 --out results/bench.json"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esf_v2 import pipeline as P
from esf_v2.esf.convert import _resample, _notch_filter, _common_average_ref
from esf_v2.esf.normalize import robust_zscore
from scipy import signal as sp

def stages(sig, labels, fs):
    t = {}; t0 = time.perf_counter()
    x = P.map_channels(sig, labels); good = ~np.all(np.isnan(x), axis=1); t["map"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if abs(fs - 250.0) > 0.5:
        gi = np.where(good)[0]; r = _resample(x[gi], fs, 250.0); y = np.full((19, r.shape[1]), np.nan, np.float32); y[gi] = r; x = y
    t["resample"] = time.perf_counter() - t0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter(); x[good] = _notch_filter(x[good], 250.0, 60.0); t["notch"] = time.perf_counter() - t0
        t0 = time.perf_counter(); sos = sp.butter(4, 0.5, btype="highpass", fs=250.0, output="sos"); x[good] = sp.sosfiltfilt(sos, x[good], axis=1).astype(np.float32); t["highpass"] = time.perf_counter() - t0
        t0 = time.perf_counter(); x = _common_average_ref(x, good); t["car"] = time.perf_counter() - t0
    t0 = time.perf_counter(); xn, _, _ = robust_zscore(x[good], sfreq=250.0); x[good] = xn; t["robust_z"] = time.perf_counter() - t0
    return np.nan_to_num(x, nan=0.0).astype(np.float32), t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tuab", type=Path, required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", type=Path, default=Path("results/bench.json"))
    a = ap.parse_args()
    import mne, torch; mne.set_log_level("ERROR")
    edfs = sorted(str(p) for p in (a.tuab / "eval").rglob("*.edf"))[: a.n]
    rows = []
    for p in edfs:
        t0 = time.perf_counter(); r = mne.io.read_raw_edf(p, preload=True, verbose=False); sig = (r.get_data()*1e6).astype(np.float32); labels = list(r.ch_names); fs = float(r.info["sfreq"]); t_read = time.perf_counter() - t0
        x, t = stages(sig, labels, fs); t["read_edf"] = t_read; t["minutes"] = sig.shape[1] / fs / 60.0; t["fs"] = fs; t["n_ch"] = sig.shape[0]
        rows.append((p, x, t))
    out = {"n_recordings": len(rows), "mean_minutes": float(np.mean([t["minutes"] for _, _, t in rows])), "contract_stage_seconds_per_recording": {}, "backbone_cpu_seconds_per_recording": {}}
    for k in ["read_edf", "map", "resample", "notch", "highpass", "car", "robust_z"]:
        v = [t[k] for _, _, t in rows]; out["contract_stage_seconds_per_recording"][k] = {"mean": float(np.mean(v)), "sd": float(np.std(v))}
    out["contract_total_seconds_per_recording"] = float(np.mean([sum(t[k] for k in ["map","resample","notch","highpass","car","robust_z"]) for _, _, t in rows]))
    out["contract_seconds_per_minute_of_eeg"] = out["contract_total_seconds_per_recording"] / out["mean_minutes"]
    from esf_v2.backbones import Backbone
    for name in ["eegpt", "labram", "biot"]:
        res = {}
        for threads in [1, os.cpu_count()]:
            torch.set_num_threads(threads); bb = Backbone(name, device="cpu"); _ = bb.window_features(rows[0][1][:, :250*60])  # warm-up
            ts = []
            for _, x, _t in rows[:10]:
                t0 = time.perf_counter(); _ = bb.window_features(x); ts.append(time.perf_counter() - t0)
            res[f"threads_{threads}"] = {"mean": float(np.mean(ts)), "sd": float(np.std(ts)), "per_minute": float(np.mean(ts) / out["mean_minutes"])}
        n_params = sum(p.numel() for p in bb.model.parameters()); res["params_M"] = n_params / 1e6; res["window_s"] = bb.win / bb.fs; res["dim"] = bb.dim
        out["backbone_cpu_seconds_per_recording"][name] = res; print(name, res, flush=True)
    a.out.parent.mkdir(exist_ok=True, parents=True); json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
