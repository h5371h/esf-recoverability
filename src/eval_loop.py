"""
SPMB perturbation sweep evaluation loop (paper §Results: degradation by axis).

Supports two tasks (selected with --task), each driving a different
corpus + head + recording-level aggregation:

  --task normal_abnormal  (default; legacy behaviour)
      Corpus : TUAB eval split (binary normal/abnormal)
      Head   : Head A v1 EEGPT (--head-a-ckpt)
      Aggregation: mean of group logits (matches Head A v1 training pooling)

  --task ied_present
      Corpus : TUEV eval split (binary IED-present)
      Head   : Head C v1 EEGPT (--head-c-ckpt)
      Aggregation: max of group logits (one IED group is enough to flag
                   the recording — matches the clinical decision rule and
                   Head C v1's positive-class definition; see
                   src/tuev_labels.py).

For each (axis, severity, arm) combination defined in
perturbations.AXES, this script:

  1. Streams each eval recording from disk (one at a time → bounded RAM).
  2. Applies the perturbation function.
  3. If canonicalized arm, applies inverse-canonicalization as defined in
     the perturbation (resample, gain-restore, notch, montage-fill, etc).
  4. Runs the *same* frozen EEGPT backbone + the task's head probe used at
     training time (apps/training/vertex/train_head_{a,c}_eegpt.py).
  5. Aggregates window logits → recording-level prediction (mean for Head A,
     max for Head C — see "Aggregation" above).
  6. Computes AUROC + balanced accuracy + ECE on the eval split with a
     bootstrap CI on 1000 resamples.
  7. Writes one CSV row per (axis, severity, arm).

Output CSV columns (identical schema across tasks for downstream re-use):
    axis, severity, arm, auroc, bal_acc, ece, n_recordings, ci_lo, ci_hi

The script also writes a per-recording predictions CSV for the
selective_prediction.py adequacy stat. Filename:
    {output-dir}/per_recording_predictions.csv
    columns: rec_id, axis, severity, arm, label, logit, prob, var_logit

Usage on VM (one tmux session, ~3-6h wall) — TUAB / Head A (default):

    PYTHONPATH=src python3 -m eval_loop \\
      --tuab-eval $TUH_ROOT/edf/eval \\
      --head-a-ckpt $CKPT_DIR/head_a_v1_eegpt.pt \\
      --eegpt-onnx $MODEL_DIR/eegpt_v1.onnx \\
      --output-dir ./data/spmb_sweep

Usage on VM — TUEV / Head C (IED present):

    PYTHONPATH=src python3 -m eval_loop \\
      --task ied_present \\
      --tuev-eval $TUH_ROOT/tuh_eeg_events/v2.0.1/edf/eval \\
      --head-c-ckpt $CKPT_DIR/head_c_v1_eegpt.pt \\
      --eegpt-onnx $MODEL_DIR/eegpt_v1.onnx \\
      --output-dir ./data/spmb_sweep_ied

Smoke mode (5 recordings × 1 severity per axis, no bootstrap):

    ... --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return float(total)


def bootstrap_auroc_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI on AUROC over `n_resamples` resamples."""
    from sklearn.metrics import roc_auc_score  # type: ignore

    n = len(labels)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_resamples, dtype=np.float64)
    aucs[:] = np.nan
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yb = labels[idx]
        pb = probs[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs[i] = roc_auc_score(yb, pb)
    aucs = aucs[~np.isnan(aucs)]
    if len(aucs) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


# ---------------------------------------------------------------------------
# Head + backbone loading
# ---------------------------------------------------------------------------
def _load_mlp_head_ckpt(ckpt_path: Path):
    """Load a 2-layer MLP head (.pt) shared by Head A v1 and Head C v1.

    Both heads use the identical architecture and checkpoint contract
    (apps/training/vertex/train_head_{a,c}_eegpt.py):
      Linear(1536, hidden) → ReLU → Dropout(p) → Linear(hidden, 1)
    Checkpoint keys: head_state, scaler_mean, scaler_scale, input_dim,
    hidden, dropout.

    Returns: (head, scaler_mean, scaler_scale, input_dim)
    """
    import torch
    import torch.nn as nn

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    input_dim = int(ckpt["input_dim"])
    hidden = int(ckpt.get("hidden", 256))
    dropout = float(ckpt.get("dropout", 0.2))
    head = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    return head, scaler_mean, scaler_scale, input_dim


# Back-compat alias — older imports may still reach for this name.
_load_head_a_v1 = _load_mlp_head_ckpt


def _load_edf_signals(edf_path: Path) -> Optional[np.ndarray]:
    """Read an EDF and return (19, n_samples) µV @ canonical 250 Hz via ESF.

    Mirrors the ingest contract used by Head A v1 training
    (apps.training.vertex.train_head_a_eegpt._load_edf_to_esf_signals).
    """
    try:
        import mne  # type: ignore
        mne.set_log_level("ERROR")
        from esf.convert import to_esf            # type: ignore
        from esf.schema import RawEEG             # type: ignore

        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        sfreq = float(raw.info["sfreq"])
        data_uv = (raw.get_data() * 1e6).astype(np.float32)

        raw_eeg = RawEEG(
            signals=data_uv,
            channel_labels=list(raw.ch_names),
            sampling_rate=sfreq,
            line_frequency=60,   # TUH source is 60 Hz mains
            source_format="edf",
            source_vendor="tuh",
        )
        esf = to_esf(raw_eeg)
        sig = np.asarray(esf.signals, dtype=np.float32)
        if sig.shape[0] != 19:
            logger.warning(f"  skip {edf_path.name}: shape {sig.shape}")
            return None
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        return sig
    except Exception as e:
        logger.warning(f"  edf load fail {edf_path.name}: {e}")
        return None


def _label_from_path(p: Path) -> Optional[int]:
    parts = [s.lower() for s in p.parts]
    if "abnormal" in parts:
        return 1
    if "normal" in parts:
        return 0
    return None


# ---------------------------------------------------------------------------
# Per-recording predict
# ---------------------------------------------------------------------------
def _predict_recording(
    signal: np.ndarray,
    backbone,
    head,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    aggregator: str = "mean",
) -> tuple[float, float]:
    """One recording → (recording_logit, window_logit_variance).

    Returns a recording-level logit and the across-group variance (used by
    selective_prediction.py as the one-class adequacy statistic).

    `aggregator`:
      "mean" — average of group logits (Head A v1: TUAB normal/abnormal;
               models population-average evidence across the recording)
      "max"  — max of group logits (Head C v1: TUEV IED-present; one
               positive-looking segment is clinical evidence of an IED).
               This matches the labelling rule used by Head C v1
               training (src/tuev_labels.py:
               "Recording-level positive iff ANY segment carries IED").
    """
    import torch

    win_feats = backbone.extract_windows(signal)
    if win_feats.shape[0] == 0:
        return (0.0, 0.0)

    # Pool per window using the same 3×512 = 1536-dim head input contract
    # used at training time (mean, max, p95 of per-window EEGPT features),
    # but applied to a SINGLE-WINDOW basis so each window gets its own
    # prediction → we can compute window-logit variance for the abstention rule.
    # Faster equivalent: feed each window's 512-d as (mean=max=p95=feat) into
    # head. To avoid that approximation, we pool over a rolling sub-set of
    # windows so the head still sees its native 1536-d input.
    #
    # Strategy: split windows into groups of size K (K = max(1, n_win//5))
    # and pool each group. Head sees 5-ish 1536-d vectors per recording;
    # window-logit-variance is computed across these groups.
    n_win = win_feats.shape[0]
    n_groups = max(1, min(8, n_win))
    group_size = max(1, n_win // n_groups)
    group_logits = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n_win
        group = win_feats[start:end]
        if group.shape[0] == 0:
            continue
        mean_f = group.mean(axis=0)
        max_f = group.max(axis=0)
        p95_f = np.percentile(group, 95.0, axis=0).astype(np.float32)
        # L2-normalize each pooled component independently (mirrors
        # apps/iplane/models/eegpt_backbone.EEGPTBackbone.extract_features).
        def _l2(v):
            n = float(np.linalg.norm(v))
            return v if n < 1e-9 else (v / n).astype(np.float32)
        feat = np.concatenate([_l2(mean_f), _l2(max_f), _l2(p95_f)]).astype(np.float32)
        # Apply the training-time StandardScaler.
        feat_s = (feat - scaler_mean) / np.where(scaler_scale > 1e-9, scaler_scale, 1.0)
        feat_t = torch.tensor(feat_s, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = float(head(feat_t).item())
        group_logits.append(logit)

    if not group_logits:
        return (0.0, 0.0)
    group_logits_arr = np.asarray(group_logits, dtype=np.float64)
    if aggregator == "max":
        rec_logit = float(group_logits_arr.max())
    elif aggregator == "mean":
        rec_logit = float(group_logits_arr.mean())
    else:
        raise ValueError(f"aggregator must be 'mean' or 'max'; got {aggregator!r}")
    var_logit = float(group_logits_arr.var(ddof=0)) if len(group_logits_arr) > 1 else 0.0
    return (rec_logit, var_logit)


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--task", type=str, default="normal_abnormal",
                    choices=["normal_abnormal", "ied_present"],
                    help="Sweep task: "
                         "normal_abnormal (default; TUAB + Head A v1, recording "
                         "aggregator=mean) or ied_present (TUEV + Head C v1, "
                         "recording aggregator=max).")
    # TUAB / Head A — required only for --task normal_abnormal (legacy default).
    ap.add_argument("--tuab-eval", type=Path, default=None,
                    help="Root of TUAB EDF eval split (required for "
                         "--task normal_abnormal), e.g. /data/tuh_corpus/edf/eval")
    ap.add_argument("--head-a-ckpt", type=Path, default=None,
                    help="Head A v1 EEGPT checkpoint (.pt) — required for "
                         "--task normal_abnormal")
    # TUEV / Head C — required only for --task ied_present.
    ap.add_argument("--tuev-eval", type=Path, default=None,
                    help="Root of TUEV EDF corpus (required for --task "
                         "ied_present), e.g. $TUH_ROOT/"
                         "tuh_eeg_events/v2.0.1/edf/eval. Recordings are "
                         "discovered by rglob *.edf; both eval/ "
                         "(filename-prefix) and train/ (.rec-companion) "
                         "layouts are supported by the same labelling "
                         "module (tuev_labels).")
    ap.add_argument("--head-c-ckpt", type=Path, default=None,
                    help="Head C v1 EEGPT checkpoint (.pt) — required for "
                         "--task ied_present")
    ap.add_argument("--eegpt-onnx", type=Path, required=True,
                    help="EEGPT v1 ONNX backbone (shared across heads)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap # eval recordings (debug)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: limit to 5 recordings, 1 severity per "
                         "axis (the middle index), bootstrap-resamples=50. "
                         "Used to verify no crashes before the long sweep; "
                         "completes in a few minutes.")
    ap.add_argument("--threads", type=int, default=8,
                    help="ORT intra-op threads")
    ap.add_argument("--backbone-batch-size", type=int, default=16,
                    help="Windows per EEGPT forward pass")
    ap.add_argument("--bootstrap-resamples", type=int, default=1000,
                    help="N bootstrap resamples for CI")
    ap.add_argument("--axis-filter", type=str, default=None,
                    help="Comma-separated axes to run (default: all five). "
                         "Names: sampling_rate, calibration, power_line, "
                         "montage, broadband")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Task wiring (corpus root, head ckpt, walker, aggregator) ────────────
    # Validate task-specific args BEFORE heavy imports so the CLI fails fast
    # in environments missing torch / sklearn / onnxruntime (e.g. CI).
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if args.task == "normal_abnormal":
        if args.tuab_eval is None or args.head_a_ckpt is None:
            ap.error("--task normal_abnormal requires --tuab-eval and --head-a-ckpt")
        corpus_root: Path = args.tuab_eval
        head_ckpt: Path = args.head_a_ckpt
        head_name = "Head A v1"
        aggregator = "mean"

        def _label_fn(p: Path):
            return _label_from_path(p)
        corpus_name = "TUAB eval"
        pos_name, neg_name = "abnormal", "normal"
    elif args.task == "ied_present":
        if args.tuev_eval is None or args.head_c_ckpt is None:
            ap.error("--task ied_present requires --tuev-eval and --head-c-ckpt")
        from tuev_labels import (   # noqa: E402
            label_from_path as _tuev_label_from_path,
        )
        corpus_root = args.tuev_eval
        head_ckpt = args.head_c_ckpt
        head_name = "Head C v1"
        aggregator = "max"

        def _label_fn(p: Path):
            return _tuev_label_from_path(p)
        corpus_name = "TUEV eval"
        pos_name, neg_name = "ied_present", "bckg_only"
    else:
        raise ValueError(f"unhandled task {args.task!r}")  # unreachable; argparse choices

    # Heavy imports — module is still importable for unit tests because main()
    # holds these inside the function body.
    from models.eegpt_backbone import EEGPTBackbone                        # noqa: E402
    from perturbations import AXES                                          # noqa: E402
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score    # noqa: E402

    # ── Smoke-mode parameter compression ────────────────────────────────────
    smoke_limit = 5
    if args.smoke:
        if args.limit is None or args.limit > smoke_limit:
            args.limit = smoke_limit
        if args.bootstrap_resamples > 50:
            args.bootstrap_resamples = 50
        logger.info(
            f"SMOKE MODE: --limit={args.limit}, --bootstrap-resamples="
            f"{args.bootstrap_resamples}, 1 severity per axis (middle index)"
        )

    backbone = EEGPTBackbone(
        args.eegpt_onnx,
        intra_op_threads=args.threads,
        batch_size=args.backbone_batch_size,
    )
    logger.info(f"Backbone info: {backbone.info}")

    head, scaler_mean, scaler_scale, input_dim = _load_mlp_head_ckpt(head_ckpt)
    if input_dim != backbone.output_dim:
        raise RuntimeError(
            f"{head_name} input_dim {input_dim} != backbone output_dim "
            f"{backbone.output_dim}"
        )
    logger.info(f"{head_name} loaded: input_dim={input_dim}  ckpt={head_ckpt}")
    logger.info(f"Task wiring: task={args.task}  aggregator={aggregator}  "
                f"corpus={corpus_name}")

    # ── Discover eval recordings ────────────────────────────────────────────
    edfs = sorted(corpus_root.rglob("*.edf"))
    samples: list[tuple[Path, int]] = []
    for edf in edfs:
        y = _label_fn(edf)
        if y is None:
            continue
        samples.append((edf, y))
    if args.limit:
        samples = samples[: args.limit]
    n_neg = sum(1 for _, y in samples if y == 0)
    n_pos = sum(1 for _, y in samples if y == 1)
    logger.info(f"{corpus_name}: {len(samples)} recordings "
                f"({neg_name}={n_neg}, {pos_name}={n_pos})")
    if not samples:
        logger.error("No labelled eval recordings found — abort.")
        return 2

    # ── Cache the raw ESF signal per recording so we don't re-decode EDFs
    # for every (axis, severity, arm). Each recording: ~10MB float32 in
    # memory; we keep them on disk in a per-rec .npy under output_dir.
    sig_cache_dir = args.output_dir / "esf_cache"
    sig_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"ESF cache → {sig_cache_dir}")

    def _esf_for(edf_path: Path) -> Optional[np.ndarray]:
        # Stable hash on the absolute path → cache filename.
        import hashlib
        h = hashlib.sha256(str(edf_path.resolve()).encode()).hexdigest()[:16]
        cache_p = sig_cache_dir / f"{edf_path.stem}_{h}.npy"
        if cache_p.exists():
            try:
                return np.load(cache_p).astype(np.float32)
            except Exception as e:
                logger.warning(f"  cache reread fail {edf_path.name}: {e}")
        sig = _load_edf_signals(edf_path)
        if sig is None:
            return None
        try:
            np.save(cache_p, sig)
        except Exception as e:
            logger.warning(f"  cache write fail {edf_path.name}: {e}")
        return sig

    # ── Axis filter ─────────────────────────────────────────────────────────
    if args.axis_filter:
        wanted = {a.strip() for a in args.axis_filter.split(",") if a.strip()}
        axes_to_run = {k: v for k, v in AXES.items() if k in wanted}
    else:
        axes_to_run = AXES
    # Smoke mode: keep only the middle severity per axis so the cross-product
    # is small. We still loop over the same axes so the no-crash guarantee
    # covers every perturbation function path.
    if args.smoke:
        smoke_axes = {}
        for k, spec in axes_to_run.items():
            sevs = spec["severities"]
            mid_idx = len(sevs) // 2
            smoke_axes[k] = {**spec, "severities": [sevs[mid_idx]]}
        axes_to_run = smoke_axes
    logger.info(f"Axes to sweep: {sorted(axes_to_run.keys())}")

    # ── Output writers ──────────────────────────────────────────────────────
    sweep_csv = args.output_dir / "sweep.csv"
    per_rec_csv = args.output_dir / "per_recording_predictions.csv"
    sweep_fh = open(sweep_csv, "w", newline="")
    per_rec_fh = open(per_rec_csv, "w", newline="")
    sweep_writer = csv.writer(sweep_fh)
    per_rec_writer = csv.writer(per_rec_fh)
    sweep_writer.writerow(
        ["axis", "severity", "arm", "auroc", "bal_acc", "ece",
         "n_recordings", "ci_lo", "ci_hi"]
    )
    per_rec_writer.writerow(
        ["rec_id", "axis", "severity", "arm", "label", "logit", "prob", "var_logit"]
    )

    sweep_t0 = time.time()
    total_evals = 0
    for axis_name, spec in axes_to_run.items():
        fn = spec["fn"]
        for sev_params in spec["severities"]:
            sev_value = sev_params[spec["severity_key"]]
            for arm in ("naive", "canonicalized"):
                t0 = time.time()
                logits_acc: list[float] = []
                labels_acc: list[int] = []
                vars_acc: list[float] = []
                rec_ids: list[str] = []
                for idx, (edf, y) in enumerate(samples, 1):
                    sig_clean = _esf_for(edf)
                    if sig_clean is None:
                        continue
                    params = dict(sev_params)
                    params["arm"] = arm
                    params["seed"] = args.seed + idx   # per-recording seed
                    try:
                        sig_pert = fn(sig_clean, params)
                    except Exception as e:
                        logger.warning(
                            f"  perturbation fail {axis_name}/{sev_value}/{arm} "
                            f"on {edf.name}: {e}; skipping recording"
                        )
                        continue

                    # If sampling-rate canonicalize resampled to 250 Hz we
                    # already have CANONICAL_FS_HZ. Naive sampling-rate arm
                    # leaves a different effective rate — the backbone still
                    # uses 1000-sample windows, which on a 128 Hz "signal"
                    # represents ~7.8s of real time. That's the silent
                    # degradation the paper measures.
                    if sig_pert.shape[0] != 19:
                        continue

                    rec_logit, var_logit = _predict_recording(
                        sig_pert, backbone, head, scaler_mean, scaler_scale,
                        aggregator=aggregator,
                    )
                    logits_acc.append(rec_logit)
                    labels_acc.append(y)
                    vars_acc.append(var_logit)
                    rec_ids.append(edf.stem)

                    if idx % 25 == 0 or idx == len(samples):
                        rate = idx / (time.time() - t0 + 1e-6)
                        logger.info(
                            f"  [{axis_name}/{sev_value}/{arm}] "
                            f"{idx}/{len(samples)} rate={rate:.2f}/s"
                        )

                if not labels_acc:
                    logger.warning(
                        f"  skip CSV row {axis_name}/{sev_value}/{arm}: 0 recordings"
                    )
                    continue
                labels_np = np.asarray(labels_acc, dtype=np.int64)
                logits_np = np.asarray(logits_acc, dtype=np.float64)
                probs_np = 1.0 / (1.0 + np.exp(-logits_np))
                yhat_np = (probs_np >= 0.5).astype(np.int64)

                if len(np.unique(labels_np)) > 1:
                    auroc = float(roc_auc_score(labels_np, probs_np))
                else:
                    auroc = float("nan")
                bal_acc = float(balanced_accuracy_score(labels_np, yhat_np))
                ece = compute_ece(probs_np, labels_np)
                ci_lo, ci_hi = bootstrap_auroc_ci(
                    labels_np, probs_np,
                    n_resamples=args.bootstrap_resamples,
                    seed=args.seed,
                )
                sweep_writer.writerow([
                    axis_name, sev_value, arm,
                    f"{auroc:.6f}", f"{bal_acc:.6f}", f"{ece:.6f}",
                    len(labels_acc), f"{ci_lo:.6f}", f"{ci_hi:.6f}",
                ])
                sweep_fh.flush()
                for rid, lab, lo, p, v in zip(rec_ids, labels_acc, logits_acc, probs_np.tolist(), vars_acc):
                    per_rec_writer.writerow([
                        rid, axis_name, sev_value, arm,
                        lab, f"{lo:.6f}", f"{p:.6f}", f"{v:.6f}",
                    ])
                per_rec_fh.flush()
                total_evals += 1
                logger.info(
                    f"  ROW {axis_name}/{sev_value}/{arm}: "
                    f"AUROC={auroc:.4f}  bal_acc={bal_acc:.4f}  ECE={ece:.4f}  "
                    f"n={len(labels_acc)}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
                    f"elapsed={(time.time()-t0)/60:.1f}min"
                )

    sweep_fh.close()
    per_rec_fh.close()
    summary = {
        "sweep_csv": str(sweep_csv),
        "per_recording_csv": str(per_rec_csv),
        "n_rows": total_evals,
        "n_recordings": len(samples),
        "wall_time_minutes": (time.time() - sweep_t0) / 60.0,
        "axes": list(axes_to_run.keys()),
        "task": args.task,
        "aggregator": aggregator,
        "head_name": head_name,
        "head_ckpt": str(head_ckpt),
        "eegpt_onnx": str(args.eegpt_onnx),
        "corpus_root": str(corpus_root),
        "corpus_name": corpus_name,
        "smoke": bool(args.smoke),
        # Back-compat fields the existing TUAB/Head A consumers may still
        # parse — these are only meaningful for task=normal_abnormal.
        "head_a_ckpt": str(args.head_a_ckpt) if args.head_a_ckpt else None,
        "tuab_eval":   str(args.tuab_eval)   if args.tuab_eval   else None,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
    }
    (args.output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote sweep CSV     {sweep_csv}")
    logger.info(f"Wrote per-rec CSV   {per_rec_csv}")
    logger.info(f"Wrote summary JSON  {args.output_dir / 'sweep_summary.json'}")
    logger.info(f"Total wall time: {summary['wall_time_minutes']:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
