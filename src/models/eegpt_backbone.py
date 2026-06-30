"""
EEGPT backbone — frozen ONNX inference wrapper used by the SPMB sweep.

Built on top of the public braindecode/eegpt-pretrained checkpoint. The
runtime contract is intentionally narrow:

  * inputs : (19, n_samples) ESF prenorm canonical, 250 Hz (the
             braindecode HF model is natively 250 Hz, so no resampling
             happens inside this wrapper).
  * output : (n_windows, 512) float32 — encoder summary tokens averaged
             across embed_num (4) and patches per window (~31).

Why ONNX:
  * No torch dependency in the runtime image, no HuggingFace hub
    download at startup, deterministic load time.
  * The wrapper loads a pre-exported ONNX session and returns the
    pooled-feature contract that the downstream linear probe consumes.

ONNX export procedure (one-shot, requires torch + braindecode):
    import torch
    from braindecode.models import EEGPT as _BD_EEGPT  # public package
    model = _BD_EEGPT.from_pretrained(
        "braindecode/eegpt-pretrained",
        n_chans=19, n_times=1000, sfreq=250.0,
    )
    model.eval()
    dummy = torch.zeros(1, 19, 1000)
    torch.onnx.export(
        model, dummy, "eegpt_v1.onnx",
        input_names=["esf_window"], output_names=["tokens"],
        dynamic_axes={"esf_window": {0: "batch"},
                      "tokens":     {0: "batch"}},
        opset_version=17,
    )

The 19-channel order in the exported ONNX is the EEGPT-native order
(EEGPT_CHANNEL_ORDER below). The wrapper reorders ESF -> EEGPT before
the session.run() call.

Stateless + thread-safe: the ONNX Runtime InferenceSession is read-only
after construction. Multiple concurrent .extract_features() calls share
the session safely under ORT's default thread model.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Frozen contract constants for the braindecode/eegpt-pretrained ──────────
# checkpoint. Do not drift these from the values used when the ONNX was
# exported — the downstream linear probe was trained against the exact
# 19-channel order and 250 Hz / 4-second window shape pinned here.
EEGPT_SFREQ = 250.0
EEGPT_N_TIMES_PER_WIN = 1000          # 4-sec window at 250 Hz (braindecode native)
EEGPT_EMBED_DIM = 512
ESF_N_CHANS = 19

ESF_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1",  "O2",  "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz",  "Cz",  "Pz",
]
EEGPT_CHANNEL_ORDER = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7",  "C3",  "CZ", "C4", "T8",
    "P7",  "P3",  "PZ", "P4", "P8",
    "O1",  "O2",
]
_ESF_TO_EEGPT_NAME = {
    "Fp1": "FP1", "Fp2": "FP2",
    "F3": "F3", "F4": "F4", "F7": "F7", "F8": "F8", "Fz": "FZ",
    "C3": "C3", "C4": "C4", "Cz": "CZ",
    "P3": "P3", "P4": "P4", "Pz": "PZ",
    "O1": "O1", "O2": "O2",
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}
ESF_TO_EEGPT_PERM = np.asarray(
    [
        ESF_CHANNEL_ORDER.index(
            {v: k for k, v in _ESF_TO_EEGPT_NAME.items()}[eegpt_name]
        )
        for eegpt_name in EEGPT_CHANNEL_ORDER
    ],
    dtype=np.int64,
)


def _window_starts(n_samples: int, win: int, stride: int) -> list[int]:
    """Half-overlap window starts (paper §7 default stride = win//2).
    Tail handling: if the last window does not align, drop it — pooling
    on a sub-window biases features for short recordings."""
    if n_samples < win:
        return []
    starts = list(range(0, n_samples - win + 1, stride))
    return starts


class EEGPTBackbone:
    """Frozen EEGPT feature extractor (ONNX, CPU-only by default).

    Usage:
        backbone = EEGPTBackbone(Path("apps/iplane/models/eegpt_v1.onnx"))
        feats = backbone.extract_features(signals_esf_prenorm)
        # feats: (512,) recording-level pooled (mean+max+p95 of per-window).

    The recording-level pooling here mirrors what
    apps/training/vertex/train_head_a_eegpt.py does at training time —
    keeping them aligned is the entire reason this wrapper exists.
    """

    def __init__(
        self,
        onnx_path: Path,
        intra_op_threads: int = 1,
        win_samples: int = EEGPT_N_TIMES_PER_WIN,
        stride_samples: Optional[int] = None,
        batch_size: int = 16,
    ):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is required for EEGPTBackbone — pip install onnxruntime"
            ) from e

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"EEGPT ONNX missing at {self.onnx_path}. Export the "
                "frozen braindecode/eegpt-pretrained checkpoint to ONNX "
                "using the snippet in the module docstring."
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.win_samples = int(win_samples)
        # Default stride = win (no overlap) for recording-level pooled features.
        # Paper §7's 50% overlap is for window-level downstream tasks where
        # boundary effects matter; pooling-across-windows is overlap-invariant
        # to first order and halves the EEGPT forward pass cost.
        self.stride_samples = int(stride_samples) if stride_samples else self.win_samples
        self.batch_size = int(batch_size)

        # ORT sessions are thread-safe for inference, but we hold a lock for
        # diagnostic safety (avoids interleaved logs when extract_features
        # is called concurrently). The lock is on log writes only — actual
        # inference runs in parallel under ORT's worker pool.
        self._log_lock = threading.Lock()

        # Probe output shape on a synthetic forward.
        dummy = np.zeros((1, ESF_N_CHANS, self.win_samples), dtype=np.float32)
        try:
            out = self.session.run([self.output_name], {self.input_name: dummy})[0]
        except Exception as e:
            raise RuntimeError(
                f"EEGPT ONNX probe failed on {self.onnx_path.name}: {e}"
            ) from e
        # Acceptable output shapes:
        #   (1, n_patches, embed_num, embed_dim) — raw braindecode return
        #   (1, embed_dim)                       — already pooled in graph
        if out.ndim == 4:
            _, n_patches, embed_num, embed_dim = out.shape
            self._raw_output_shape = (n_patches, embed_num, embed_dim)
            self._needs_internal_pool = True
        elif out.ndim == 2:
            _, embed_dim = out.shape
            self._raw_output_shape = (1, 1, embed_dim)
            self._needs_internal_pool = False
        else:
            raise RuntimeError(
                f"EEGPT ONNX produced unexpected output rank {out.ndim} "
                f"(shape {out.shape}); expected 2 or 4."
            )
        if embed_dim != EEGPT_EMBED_DIM:
            raise RuntimeError(
                f"EEGPT embed_dim mismatch: graph emits {embed_dim}, "
                f"contract expects {EEGPT_EMBED_DIM}."
            )

        logger.info(
            f"EEGPTBackbone loaded from {self.onnx_path.name}: "
            f"raw_output_shape={self._raw_output_shape}, "
            f"needs_internal_pool={self._needs_internal_pool}, "
            f"win={self.win_samples} stride={self.stride_samples}"
        )

    # ── per-window forward ─────────────────────────────────────────────────
    def _per_window_features(self, window_esf: np.ndarray) -> np.ndarray:
        """One window → (512,) pooled across patches+embed_num.

        window_esf: (19, win_samples) float32 ESF prenorm.
        """
        if window_esf.shape != (ESF_N_CHANS, self.win_samples):
            raise ValueError(
                f"window_esf shape {window_esf.shape} != "
                f"({ESF_N_CHANS}, {self.win_samples})"
            )
        # ESF → EEGPT channel reorder.
        x = window_esf[ESF_TO_EEGPT_PERM, :]
        x = x.astype(np.float32, copy=False)[np.newaxis, :, :]   # (1, 19, win)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        if self._needs_internal_pool:
            # (1, n_patches, embed_num, embed_dim) → (embed_dim,)
            feat = out.mean(axis=(1, 2))[0]
        else:
            feat = out[0]
        # L2-normalise per spec (paper §7.2 — normalised embeddings make
        # downstream linear heads scale-invariant to recording amplitude).
        norm = float(np.linalg.norm(feat))
        if norm > 1e-9:
            feat = feat / norm
        return feat.astype(np.float32)

    def extract_windows(self, signals_esf: np.ndarray) -> np.ndarray:
        """All windows for a recording → (n_windows, 512) float32.

        Batches `batch_size` windows per ONNX call — measured 2-3× faster
        than per-window calls on E64as_v5 (66ms/window @ batch 16 vs 140ms
        single). Crucial for keeping per-recording cost under 1 min.

        signals_esf: (19, n_samples) ESF prenorm canonical. Caller is
        responsible for prenorm (per-channel robust z-score per ESF v1.1).
        """
        if signals_esf.ndim != 2 or signals_esf.shape[0] != ESF_N_CHANS:
            raise ValueError(
                f"signals_esf shape {signals_esf.shape} bad; need (19, n_samples)"
            )
        starts = _window_starts(signals_esf.shape[1], self.win_samples, self.stride_samples)
        if not starts:
            return np.zeros((0, EEGPT_EMBED_DIM), dtype=np.float32)
        # ESF → EEGPT channel reorder once per recording (cheap).
        sig_eegpt = signals_esf[ESF_TO_EEGPT_PERM, :].astype(np.float32, copy=False)
        feats = np.empty((len(starts), EEGPT_EMBED_DIM), dtype=np.float32)
        for i in range(0, len(starts), self.batch_size):
            batch_starts = starts[i : i + self.batch_size]
            batch = np.stack(
                [sig_eegpt[:, s : s + self.win_samples] for s in batch_starts],
                axis=0,
            ).astype(np.float32)
            out = self.session.run([self.output_name], {self.input_name: batch})[0]
            if self._needs_internal_pool:
                pooled = out.mean(axis=(1, 2))               # (B, embed_dim)
            else:
                pooled = out                                 # already (B, embed_dim)
            # L2-normalise per window (matches _per_window_features semantics).
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            pooled = (pooled / norms).astype(np.float32)
            feats[i : i + len(batch_starts)] = pooled
        return feats

    def extract_features(self, signals_esf: np.ndarray) -> np.ndarray:
        """Recording-level (1536,) pooled features.

        Pooling per paper §7.5 head-input spec:
          [ mean | max | p95 ] of per-window EEGPT pooled embeddings →
          concatenated to (3 * 512,) = 1536-dim head input.

        Returns zeros if signals_esf is too short for one window.
        """
        win_feats = self.extract_windows(signals_esf)
        if win_feats.shape[0] == 0:
            return np.zeros((3 * EEGPT_EMBED_DIM,), dtype=np.float32)
        mean_feat = win_feats.mean(axis=0)
        max_feat = win_feats.max(axis=0)
        # numpy.percentile takes (q, axis) — pool across windows axis 0.
        p95_feat = np.percentile(win_feats, 95.0, axis=0).astype(np.float32)
        # Re-normalise each pooled vector independently — keeps the three
        # statistics on comparable scales for the linear head.
        def _l2(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            return v if n < 1e-9 else (v / n).astype(np.float32)
        return np.concatenate([_l2(mean_feat), _l2(max_feat), _l2(p95_feat)]).astype(np.float32)

    @property
    def output_dim(self) -> int:
        """Recording-level pooled-feature dim consumed by downstream heads."""
        return 3 * EEGPT_EMBED_DIM

    @property
    def info(self) -> dict:
        return {
            "onnx_path":   str(self.onnx_path),
            "win_samples": self.win_samples,
            "stride":      self.stride_samples,
            "batch_size":  self.batch_size,
            "embed_dim":   EEGPT_EMBED_DIM,
            "output_dim":  self.output_dim,
            "needs_internal_pool": self._needs_internal_pool,
        }
