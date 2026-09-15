"""Three frozen pretrained EEG backbones behind one interface.
Each adapter takes the canonical ESF signal (19 ch, 250 Hz, normalised) and returns per-window, L2-normalised features.
The adapter (resampling to the backbone's native rate, channel selection/derivation) is a fixed, model-side step
applied identically to every arm and cell; it is not part of the acquisition operators.
  eegpt : braindecode/eegpt-pretrained, 250 Hz, 4 s windows (1000 samples), encoder tokens averaged -> 512-d
  labram: braindecode/labram-pretrained, 200 Hz, 15 s windows (3000 samples = 15 patches, the pretrained config), patch tokens averaged -> 200-d
  biot  : braindecode/biot-pretrained-prest-16chs, 200 Hz, 5 s windows (pretrained n_times), 16 TCP bipolar derivations, encoder embedding -> 256-d
"""
from __future__ import annotations
import numpy as np, torch
from scipy import signal as sp
from .esf.channels import STANDARD_CHANNELS
from .backbone import ESF_TO_EEGPT_PERM, EEGPT_CHANNEL_ORDER, WIN as EEGPT_WIN

ESF_UP = [c.upper() for c in STANDARD_CHANNELS]
_ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
LABRAM_NAMES = [_ALIAS.get(c, c) for c in ESF_UP]           # LaBraM position embeddings are looked up by name
TCP16 = [("FP1","F7"),("F7","T3"),("T3","T5"),("T5","O1"),("FP2","F8"),("F8","T4"),("T4","T6"),("T6","O2"),
         ("FP1","F3"),("F3","C3"),("C3","P3"),("P3","O1"),("FP2","F4"),("F4","C4"),("C4","P4"),("P4","O2")]
TCP16_IDX = [(ESF_UP.index(a), ESF_UP.index(b)) for a, b in TCP16]

class Backbone:
    def __init__(self, name: str, device: str = "cuda", seed: int = 7):
        self.name = name; self.device = device
        torch.manual_seed(seed); np.random.seed(seed)
        if name == "eegpt":
            from .backbone import load_backbone
            self.model, self.info = load_backbone(device, seed); self.fs = 250.0; self.win = EEGPT_WIN; self.dim = 512
        elif name == "labram":
            from braindecode.models import Labram
            m = Labram.from_pretrained("braindecode/labram-pretrained"); m.eval().to(device)
            for p in m.parameters(): p.requires_grad = False
            self.model = m; self.fs = 200.0; self.win = int(m.n_times); self.dim = int(getattr(m, "embed_dim", 200)); self.info = {"repo": "braindecode/labram-pretrained", "win": self.win, "note": "15 s windows (pretrained n_times=3000 at 200 Hz), 19 named channels, patch tokens averaged"}
        elif name == "biot":
            from braindecode.models import BIOT
            m = BIOT.from_pretrained("braindecode/biot-pretrained-prest-16chs"); m.eval().to(device)
            for p in m.parameters(): p.requires_grad = False
            self.model = m; self.fs = 200.0; self.win = int(m.n_times); self.dim = 256; self.info = {"repo": "braindecode/biot-pretrained-prest-16chs", "win": self.win, "note": "5 s windows, 16 TCP bipolar derivations, encoder embedding"}
        else:
            raise ValueError(name)

    def _prepare(self, sig_esf: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(sig_esf, nan=0.0).astype(np.float32)
        if self.fs != 250.0:
            x = sp.resample_poly(x.astype(np.float64), up=int(self.fs), down=250, axis=1).astype(np.float32)  # 4/5 exactly
        if self.name == "biot":
            x = np.stack([x[a] - x[b] for a, b in TCP16_IDX]).astype(np.float32)
        return x

    @torch.no_grad()
    def _forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.name == "eegpt":
            out = self.model(w[:, ESF_TO_EEGPT_PERM, :])
            return out.mean(dim=(1, 2)) if out.ndim == 4 else out
        if self.name == "labram":
            out = self.model(w, ch_names=LABRAM_NAMES, return_features=True)
            feats = out["features"] if isinstance(out, dict) else out
            return feats.mean(dim=1) if feats.ndim == 3 else feats
        if self.name == "biot":
            enc = getattr(self.model, "encoder", None)
            if enc is not None: return enc(w)
            self.model.return_feature = True; out = self.model(w)
            return out[-1] if isinstance(out, (tuple, list)) else out

    @torch.no_grad()
    def window_features(self, sig_esf: np.ndarray, batch: int = 128) -> np.ndarray:
        x = self._prepare(sig_esf); n = x.shape[1]; starts = list(range(0, n - self.win + 1, self.win))
        if not starts: return np.zeros((0, self.dim), np.float32)
        feats = []
        for i in range(0, len(starts), batch):
            w = torch.from_numpy(np.stack([x[:, s:s+self.win] for s in starts[i:i+batch]])).to(self.device)
            out = self._forward(w).float()
            out = out / out.norm(dim=1, keepdim=True).clamp_min(1e-9)
            feats.append(out.cpu().numpy())
        f = np.concatenate(feats, 0); self.dim = f.shape[1]; return f
