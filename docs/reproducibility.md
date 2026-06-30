# Reproducibility walkthrough

This walkthrough takes a reviewer from a bare laptop to bit-exact paper
figures.

## Tier A — five-minute reproduction (no TUH access needed)

Goal: regenerate every figure (except Magna Fig. 5/11/13) and every
statistic in `advanced_stats_summary.md` from the CSVs already in
`data/`.

```bash
git clone https://github.com/h5371h/esf-recoverability.git
cd esf-recoverability

python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Regenerate every figure
python src/paper_figures.py

# Recompute every statistical-test result
python src/run_all_advanced_stats.py

# Confirm the unit-test suite passes
pytest src/test_*.py -q
```

Expected outcome:

* `figures/fig_*.pdf` — 10 PDFs byte-different but visually identical
  to the submitted versions (matplotlib non-determinism in metadata
  timestamps).
* `data/advanced_stats_summary.md` — every number in the
  "verification" block matches the paper text to ≤ 0.005 abs.

## Tier B — full reproduction from raw TUH EDFs (~1 day)

Goal: re-run the perturbation sweep from raw EDFs and confirm the
shipped `sweep_latest.csv` is reproduced exactly.

### Prerequisites

* TUH-EEG access (NEDC application — see `data/README.md` for the
  link). About 1–2 days of human turnaround.
* Linux box or M-series Mac with ≥ 16 CPU cores and ≥ 32 GB RAM.
  Empirically the sweep takes 3–6 h on 16 cores; double that on 8 cores.
* The frozen EEGPT ONNX weights, exported once from the public
  HuggingFace `braindecode/eegpt-pretrained` checkpoint via the
  `torch.onnx.export` call in `src/models/eegpt_backbone.py`
  (module docstring). The export is a single-pass conversion — no
  training.

### Retraining Head A from scratch

Head A v1 is a single hidden-layer MLP probe on top of pooled EEGPT
features. The full training script is part of the ESF Reference Implementation private
monorepo, but a reproducible substitute is short enough to drop in here:

```python
# scripts/retrain_head_a.py (template, ~30 min wall on 16 cores)
import torch, numpy as np, glob
from src.models.eegpt_backbone import EEGPTBackbone
from src.esf.convert import to_esf
from src.esf.schema import RawEEG
import mne, torch.nn as nn

backbone = EEGPTBackbone("path/to/eegpt_v1.onnx")
xs, ys = [], []
for edf in glob.glob("$TUH_ROOT/edf/train/**/*.edf", recursive=True):
    raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
    sig = to_esf(RawEEG(
        signals=(raw.get_data() * 1e6).astype("float32"),
        channel_labels=raw.ch_names,
        sampling_rate=float(raw.info["sfreq"]),
        line_frequency=60,
    )).signals
    feats = backbone.extract_windows(sig)  # (n_win, 512)
    xs.append(feats.mean(axis=0))          # recording-level mean
    ys.append(1 if "abnormal" in edf else 0)

X = np.stack(xs); y = np.asarray(ys)
mu, sd = X.mean(0), X.std(0) + 1e-6
Xn = (X - mu) / sd
torch.manual_seed(7)
mlp = nn.Sequential(nn.Linear(512, 64), nn.ReLU(),
                    nn.Dropout(0.1), nn.Linear(64, 1))
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
xb = torch.from_numpy(Xn).float()
yb = torch.from_numpy(y).float().unsqueeze(1)
for epoch in range(200):
    opt.zero_grad()
    loss = nn.functional.binary_cross_entropy_with_logits(mlp(xb), yb)
    loss.backward(); opt.step()
torch.save({
    "head_state": mlp.state_dict(),
    "scaler_mean": mu.tolist(),
    "scaler_scale": sd.tolist(),
    "input_dim": 512,
    "hidden": 64,
    "dropout": 0.1,
}, "head_a_v1_eegpt.pt")
```

This produces the same `.pt` schema that `src/eval_loop.py` expects
(see `_load_mlp_head_ckpt`).

### Running the sweep

```bash
export TUH_ROOT=/path/to/tuh_corpus
export CKPT_DIR=/path/to/checkpoints
export MODEL_DIR=/path/to/onnx_models

PYTHONPATH=src python -m eval_loop \
  --tuab-eval $TUH_ROOT/edf/eval \
  --head-a-ckpt $CKPT_DIR/head_a_v1_eegpt.pt \
  --eegpt-onnx $MODEL_DIR/eegpt_v1.onnx \
  --output-dir ./data/spmb_sweep \
  --threads 16 \
  --bootstrap-resamples 1000 \
  --seed 7
```

This will produce two CSVs:

* `data/spmb_sweep/sweep.csv` — should match `data/sweep_latest.csv`
  row-for-row.
* `data/spmb_sweep/per_recording_predictions.csv` — should match
  `data/per_recording_predictions_latest.csv` row-for-row (logits agree
  to float32 precision; AUROC ties resolved deterministically by
  mid-rank).

### Running the statistical tests

```bash
PYTHONPATH=src python -m statistical_tests \
  --per-rec-csv ./data/spmb_sweep/per_recording_predictions.csv \
  --output-csv  ./data/spmb_sweep/statistical_tests.csv \
  --seed 7 \
  --n-resamples 1000

python src/run_all_advanced_stats.py \
  --per-rec-csv ./data/spmb_sweep/per_recording_predictions.csv \
  --sweep-csv   ./data/spmb_sweep/sweep.csv
```

## Determinism contract

| Component | Source of randomness | Seed |
| --- | --- | --- |
| Bootstrap AUROC CIs (`eval_loop.py`) | resample indices | 7 |
| Paired bootstrap on AUROC delta (`statistical_tests.py`) | resample indices | 7 |
| Mixed-effects GLMM init (`advanced_stats.py`) | statsmodels init | 7 |
| Permutation test (10 000 perms) | label permutations | 7 |
| Conformal split (`advanced_stats.py`) | calibration-set split | 7 |
| Mutual-information estimator | k-NN resampling | 7 |

Anything not in the table is fully deterministic (no randomness).

## Known floating-point caveats

* EEGPT ONNX inference on x86 vs ARM produces logits agreeing to ~ 1e-5
  absolute; AUROC and rank-order are unchanged.
* `scipy.signal.iirfilter` (notch path inside ESF canonicalization)
  uses slightly different bilinear coefficients across scipy minor
  versions ≥ 1.13; rounding-level differences in the 4th decimal of
  per-recording logits are expected. The aggregate AUROC remains
  stable to 3 decimals.
