"""Stage 1: canonical ESF + frozen EEGPT window features for every TUAB recording.
Writes features/<split>/<rec_id>.npy  (n_win, 512)  and features/<split>_index.csv (rec_id,label,path,fs,n_win).
CPU workers do EDF read + ESF; the GPU does EEGPT in batches."""
from __future__ import annotations
import argparse, csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esf_v2 import pipeline as P

def load_edf(path: str):
    import mne; mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    return (raw.get_data() * 1e6).astype(np.float32), list(raw.ch_names), float(raw.info["sfreq"])

def label_of(p: str) -> int:
    return 1 if "/abnormal/" in p else 0

def esf_job(path: str):
    try:
        sig, labels, fs = load_edf(path)
        x = P.canonical(sig, labels, fs, 60.0)
        return path, fs, x
    except Exception as e:
        return path, None, repr(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuab", type=Path, required=True, help=".../tuh_eeg_abnormal/v3.0.1/edf")
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--backbone", type=str, default="eegpt")
    a = ap.parse_args()
    if a.out is None: a.out = Path(('features' if 'features' in __file__ else 'cells') + '_' + a.backbone)
    from esf_v2.backbones import Backbone
    bb = Backbone(a.backbone, "cuda"); print("backbone:", a.backbone, bb.info, flush=True)
    edfs = sorted(str(p) for p in (a.tuab / a.split).rglob("*.edf"))
    if a.limit: edfs = edfs[: a.limit]
    outdir = a.out / a.split; outdir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in edfs if not (outdir / (Path(p).stem + ".npy")).exists()]
    print(f"{a.split}: {len(edfs)} EDFs, {len(todo)} to do", flush=True)
    rows, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(esf_job, p) for p in todo]
        for f in as_completed(futs):
            path, fs, x = f.result(); done += 1
            if fs is None:
                print(f"  FAIL {Path(path).name}: {x}", flush=True); continue
            feats = bb.window_features(x)
            np.save(outdir / (Path(path).stem + ".npy"), feats)
            rows.append((Path(path).stem, label_of(path), path, fs, feats.shape[0]))
            if done % 100 == 0: print(f"  {done}/{len(todo)}  {done/(time.time()-t0):.2f}/s", flush=True)
    # rebuild full index from disk (covers previously cached files)
    idx = a.out / f"{a.split}_index.csv"
    with open(idx, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rec_id", "label", "path", "fs", "n_win"])
        for p in edfs:
            f = outdir / (Path(p).stem + ".npy")
            if f.exists():
                n = np.load(f, mmap_mode="r").shape[0]
                fs = next((r[3] for r in rows if r[0] == Path(p).stem), float("nan"))
                w.writerow([Path(p).stem, label_of(p), p, fs, n])
    print("index:", idx, flush=True)

if __name__ == "__main__":
    main()
