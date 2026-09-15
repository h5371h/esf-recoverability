"""Frozen EEGPT backbone (torch, GPU) reproducing the SPMB v1 ONNX wrapper contract:
ESF->EEGPT channel reorder, 1000-sample windows at 250 Hz, stride = window, encoder
output averaged over (patches, embed_num) -> 512-d, L2-normalised per window.
The 19->19 chan_proj adapter is not in the public checkpoint; it is initialised with a
fixed seed and frozen, exactly as documented in v2 protocol.md."""
from __future__ import annotations
import numpy as np, torch
from .esf.channels import STANDARD_CHANNELS

EEGPT_SFREQ = 250.0
WIN = 1000
EMBED = 512
EEGPT_CHANNEL_ORDER = ["FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4","T8","P7","P3","PZ","P4","P8","O1","O2"]
_ESF_TO_EEGPT = {"Fp1":"FP1","Fp2":"FP2","F3":"F3","F4":"F4","F7":"F7","F8":"F8","Fz":"FZ","C3":"C3","C4":"C4","Cz":"CZ",
                 "P3":"P3","P4":"P4","Pz":"PZ","O1":"O1","O2":"O2","T3":"T7","T4":"T8","T5":"P7","T6":"P8"}
_EEGPT_TO_ESF = {v: k for k, v in _ESF_TO_EEGPT.items()}
ESF_TO_EEGPT_PERM = np.asarray([STANDARD_CHANNELS.index(_EEGPT_TO_ESF[n]) for n in EEGPT_CHANNEL_ORDER], dtype=np.int64)

def load_backbone(device: str = "cuda", seed: int = 7):
    from braindecode.models import EEGPT
    from huggingface_hub import hf_hub_download
    import safetensors.torch
    torch.manual_seed(seed); np.random.seed(seed)
    model = EEGPT(n_outputs=1, n_chans=19, n_times=WIN, sfreq=EEGPT_SFREQ,
                  return_encoder_output=True, chan_proj_type="conv1d_constraint", n_chans_target=19)
    sd = safetensors.torch.load_file(hf_hub_download("braindecode/eegpt-pretrained", "model.safetensors"))
    sd = {k: v for k, v in sd.items() if not (k.startswith("chan_proj.") or k == "chans_id")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    for p in model.parameters(): p.requires_grad = False
    with torch.no_grad():
        out = model(torch.zeros(1, 19, WIN, device=device))
    return model, {"missing": list(missing), "unexpected": list(unexpected), "out_shape": tuple(out.shape), "chan_proj_seed": seed}

@torch.no_grad()
def window_features(model, sig_esf: np.ndarray, device: str = "cuda", batch: int = 256) -> np.ndarray:
    """sig_esf: (19, n) float32 normalised ESF (NaN -> 0). Returns (n_win, 512) L2-normalised."""
    x = np.nan_to_num(sig_esf, nan=0.0).astype(np.float32)
    n = x.shape[1]; starts = list(range(0, n - WIN + 1, WIN))
    if not starts: return np.zeros((0, EMBED), np.float32)
    feats = []
    for i in range(0, len(starts), batch):
        w = np.stack([x[ESF_TO_EEGPT_PERM, s:s+WIN] for s in starts[i:i+batch]])  # (b,19,1000) EEGPT order
        t = torch.from_numpy(w).to(device)
        out = model(t)
        if out.ndim == 4: out = out.mean(dim=(1, 2))
        out = out / out.norm(dim=1, keepdim=True).clamp_min(1e-9)
        feats.append(out.float().cpu().numpy())
    return np.concatenate(feats, 0)

def pool_groups(win_feats: np.ndarray):
    """v1 contract: up to 8 groups of consecutive windows; per group [l2(mean)|l2(max)|l2(p95)] -> 1536."""
    n = win_feats.shape[0]
    if n == 0: return np.zeros((0, 3*EMBED), np.float32)
    n_groups = max(1, min(8, n)); gs = max(1, n // n_groups); out = []
    def l2(v):
        s = float(np.linalg.norm(v)); return v if s < 1e-9 else v / s
    for g in range(n_groups):
        a = g*gs; b = (g+1)*gs if g < n_groups-1 else n
        grp = win_feats[a:b]
        if grp.shape[0] == 0: continue
        out.append(np.concatenate([l2(grp.mean(0)), l2(grp.max(0)), l2(np.percentile(grp, 95, axis=0))]).astype(np.float32))
    return np.stack(out)

def pool_recording(win_feats: np.ndarray) -> np.ndarray:
    """Training-time contract (train_head_a_eegpt.py): one 1536-d vector over ALL windows."""
    g = win_feats
    def l2(v):
        s = float(np.linalg.norm(v)); return v if s < 1e-9 else v / s
    return np.concatenate([l2(g.mean(0)), l2(g.max(0)), l2(np.percentile(g, 95, axis=0))]).astype(np.float32)
