"""Stage 3a (v2.1): cache frozen-backbone WINDOW FEATURES for every (recording, axis, severity, arm) cell so that any
number of probes can be scored without re-running the backbone. cells/<axis>/<sev>/<arm>/<rec_id>.npy (float16, n_win x 512)."""
from __future__ import annotations
import argparse, csv, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esf_v2 import pipeline as P
from esf_v2.acquisition import AXES, NAIVE_SWITCH
from esf_v2.backbones import Backbone

def _raw(path):
    import mne; mne.set_log_level("ERROR")
    r = mne.io.read_raw_edf(path, preload=True, verbose=False)
    return (r.get_data()*1e6).astype(np.float32), list(r.ch_names), float(r.info["sfreq"])

def cell_job(path, axis, sev, arm, fixed):
    try:
        sig, labels, fs = _raw(path); spec = AXES[axis]
        rng = np.random.default_rng(zlib.crc32(f"{Path(path).stem}|{axis}|{sev}".encode()))
        sig2, labels2, fs2, lf2 = spec["fn"](sig, labels, fs, 60.0, sev, rng=rng)
        kw = dict(NAIVE_SWITCH[axis]) if arm == "naive" else {}
        if kw.get("norm") == "fixed": kw["fixed_scale"] = fixed
        return (path, P.run(sig2, labels2, fs2, lf2, **kw))
    except Exception as e:
        return (path, repr(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuab", type=Path, required=True); ap.add_argument("--head", type=Path, required=True, help="for fixed-scale constants")
    ap.add_argument("--out", type=Path, default=None); ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--axes", type=str, default=None); ap.add_argument("--arms", type=str, default="naive,canonicalized"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--backbone", type=str, default="eegpt")
    a = ap.parse_args()
    if a.out is None: a.out = Path(('features' if 'features' in __file__ else 'cells') + '_' + a.backbone)
    import torch; ck = torch.load(a.head, weights_only=False); fixed = (ck["fixed_med"], ck["fixed_iqr"])
    bb = Backbone(a.backbone, "cuda"); print("backbone:", a.backbone, bb.info, flush=True)
    edfs = sorted(str(p) for p in (a.tuab / "eval").rglob("*.edf"))
    if a.limit: edfs = edfs[: a.limit]
    axes = a.axes.split(",") if a.axes else [k for k in AXES if k != "bandwidth"]
    arms = a.arms.split(",")
    cells = [(ax, sev, arm) for ax in axes for sev in AXES[ax]["severities"] for arm in arms]
    t0 = time.time(); n = 0
    for ax, sev, arm in cells:
        sev_s = AXES[ax]["fmt"](sev); d = a.out / ax / sev_s / arm; d.mkdir(parents=True, exist_ok=True)
        todo = [p for p in edfs if not (d / (Path(p).stem + ".npy")).exists()]
        if not todo: continue
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(cell_job, p, ax, sev, arm, fixed) for p in todo]
            for f in as_completed(futs):
                path, x = f.result()
                if isinstance(x, str): print(f"  FAIL {ax}/{sev_s}/{arm} {Path(path).name}: {x}", flush=True); continue
                wf = bb.window_features(x); np.save(d / (Path(path).stem + ".npy"), wf.astype(np.float16)); n += 1
        print(f"cell {ax}/{sev_s}/{arm} cached ({n} recs, {time.time()-t0:.0f}s)", flush=True)
    with open(a.out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rec_id", "label"]); [w.writerow([Path(p).stem, 1 if "/abnormal/" in p else 0]) for p in edfs]
    print("DONE_CELLS", flush=True)

if __name__ == "__main__":
    main()
