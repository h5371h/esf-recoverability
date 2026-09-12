# esf-recoverability

Reference implementation and reproducibility artefacts for:

> **Dammu, H.** "ESF: A Canonical Representation and Recoverability
> Framework for Deploying Pretrained EEG Classifiers in Heterogeneous
> Clinics."
> *IEEE Signal Processing in Medicine and Biology Symposium (SPMB)*,
> 2026.

This repository lets a reviewer or independent researcher reproduce
every table, figure, and statistical claim in the paper from the
same TUAB recordings + the same RNG seed.

---

## What is reproducible from a fresh clone

| Artifact | Source in repo | Effort to regenerate |
| --- | --- | --- |
| Table 1: five-axis sweep (46 cells) | `data/sweep_tuab_final.csv` | already shipped |
| Per-recording predictions, five-axis run (12 696 rows) | `data/per_recording_predictions_tuab_final.csv` | already shipped |
| Three-axis run used by Figures 2-5 (28 cells, 7 728 rows) | `data/sweep_latest.csv`, `data/per_recording_predictions_latest.csv` | already shipped |
| Figures 1, 2, 3, 5 (PDF) | `src/paper_figures.py` | `python src/paper_figures.py` (~30 s) |
| Figure 4 + Table 2 (failure taxonomy) | `src/failure_taxonomy.py` -> `figures/fig_failure_taxonomy.pdf`, `data/per_recording_failure_profile.csv` | `python src/failure_taxonomy.py` (~20 s) |
| Pinsker bound validation (Sec. IV-C) | `src/validation_compute.py` -> `data/theory/` | `python src/validation_compute.py` (~10 s) |
| Advanced statistics (FDR, mixed-effects, AURC, ...) | `data/advanced_stats_results.json` + `.md` | `python src/run_all_advanced_stats.py` (~1 min) |
| End-to-end sweep from raw EDFs | `src/eval_loop.py` | requires TUH access; ~3-6 h on 16-core CPU |

`figures/captions.md` states, figure by figure, which of the two data
runs each panel and table comes from. No clinical recordings are used in
the paper or in this repository.

---

## Citation

Once the paper is accepted, the canonical BibTeX entry will be:

```bibtex
@inproceedings{dammu2026esf,
  author    = {Hitesh Dammu},
  title     = {{ESF}: A Canonical Representation and Recoverability
               Framework for Deploying Pretrained {EEG} Classifiers in
               Heterogeneous Clinics},
  booktitle = {IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
  year      = {2026},
}
```

If you use the code or the per-recording predictions, please also cite
the archived software release (tag `v1.0.0-spmb2026-submission`):

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22726841.svg)](https://doi.org/10.5281/zenodo.22726841)

```bibtex
@software{dammu2026_esf_recoverability,
  author    = {Hitesh Dammu},
  title     = {esf-recoverability: reference implementation and
               per-recording results for IEEE SPMB 2026},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0-spmb2026-submission},
  doi       = {10.5281/zenodo.22726841},
  url       = {https://doi.org/10.5281/zenodo.22726841},
}
```

---

## Quickstart (reproduce figures + stats from shipped CSVs, no TUH needed)

```bash
git clone https://github.com/h5371h/esf-recoverability.git
cd esf-recoverability

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Regenerate the figures, the failure taxonomy (Fig. 4 + Table 2) and the Pinsker validation
python src/paper_figures.py
python src/failure_taxonomy.py
python src/validation_compute.py
ls figures/*.pdf data/theory/

# 2) Recompute all advanced statistics + verification block
python src/run_all_advanced_stats.py
cat data/advanced_stats_summary.md

# 3) Run the unit tests (no real data needed)
pytest src/test_*.py -q
```

Expected wall time: under 5 minutes on a laptop. Numeric outputs match
the values printed in the paper to ≤ 0.005 absolute AUROC.

---

## Full reproduction (from raw TUH EDFs)

End-to-end this takes about a day, dominated by waiting on the NEDC
licence approval and the EEGPT forward passes.

### Step 1 — Obtain TUH-EEG access (~1–2 days, free)

1. Visit https://isip.piconepress.com/projects/nedc/html/forms/dat_use.shtml
   and accept the NEDC data-use agreement (institutional email
   required).
2. Wait for the rsync credentials by email.
3. Fetch only the TUAB v3.0.1 eval split (the 276 recordings used in
   the paper):

   ```bash
   export TUH_USER=...  # supplied by NEDC
   rsync -auxvL "${TUH_USER}@www.isip.piconepress.com:~/data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf/eval/" "$TUH_ROOT/edf/eval/"
   ```

   Approx 8 GB on disk. See `docs/corpus_setup.md` for the full set
   of subcorpora the paper touches.

### Step 2 — Train Head A v1 (~30 min on a 16-core box)

Head A v1 is the EEGPT linear probe used by the sweep. The training
script lives in the ESF Reference Implementation private monorepo because it is a
product component; **the public substitute is to retrain the linear
probe from EEGPT features in five lines.** A reference script lives in
`docs/reproducibility.md` (section "Retraining Head A from scratch").

The frozen EEGPT ONNX checkpoint must be exported once from the
public `braindecode/eegpt-pretrained` HuggingFace repo — see the
exact `torch.onnx.export` invocation in
`src/models/eegpt_backbone.py` (module docstring).

### Step 3 — Run the perturbation sweep (~3–6 h on 16-core CPU)

```bash
export TUH_ROOT=/path/to/tuh_corpus     # parent of edf/eval/
export CKPT_DIR=/path/to/checkpoints    # holds head_a_v1_eegpt.pt
export MODEL_DIR=/path/to/onnx_models   # holds eegpt_v1.onnx

PYTHONPATH=src python -m eval_loop \
  --tuab-eval $TUH_ROOT/edf/eval \
  --head-a-ckpt $CKPT_DIR/head_a_v1_eegpt.pt \
  --eegpt-onnx $MODEL_DIR/eegpt_v1.onnx \
  --output-dir ./data/spmb_sweep
```

This writes `data/spmb_sweep/sweep.csv` and
`data/spmb_sweep/per_recording_predictions.csv` — the exact two
artefacts shipped under `data/*_latest.csv`.

### Step 4 — Statistical tests + figures (~5 min)

```bash
# Per-(axis, severity) DeLong + paired-bootstrap delta-AUROC
PYTHONPATH=src python -m statistical_tests \
  --per-rec-csv ./data/spmb_sweep/per_recording_predictions.csv \
  --output-csv  ./data/statistical_tests.csv

# Full statistical battery + paper-number verification
python src/run_all_advanced_stats.py \
  --per-rec-csv ./data/spmb_sweep/per_recording_predictions.csv \
  --sweep-csv   ./data/spmb_sweep/sweep.csv

# Regenerate every figure
python src/paper_figures.py
```

---

## Layout

```
esf-recoverability/
  README.md                           ← this file
  LICENSE                             ← MIT
  CITATION.cff                        ← machine-readable cite + Zenodo hook
  requirements.txt                    ← pinned deps for fresh checkout
  data/
    sweep_latest.csv                  ← aggregate (28 rows, no PII)
    per_recording_predictions_latest.csv (276 rec × 28 conds = 7 728 rows)
    manifest_tuab_eval.json           ← 276 TUH rec_ids used
    seeds.json                        ← every RNG seed (default = 7)
    advanced_stats_results.json
    advanced_stats_summary.md
    README.md                         ← how to get TUH yourself
  src/
    perturbations.py                  ← the 5 acquisition-shift axes
    eval_loop.py                      ← TUH eval CLI
    statistical_tests.py              ← DeLong + paired-bootstrap
    advanced_stats.py                 ← 12-method stats library
    run_all_advanced_stats.py         ← orchestrator → JSON + MD
    selective_prediction.py           ← coverage-risk
    tuev_labels.py                    ← (optional) TUEV / IED head
    paper_figures.py                  ← every figure, pure CSV-in PDF-out
    test_perturbations.py             ← unit tests
    test_statistical_tests.py
    test_tuev_labels.py
    esf/                              ← canonicalization library
      channels.py  convert.py
      schema.py    normalize.py
    models/
      eegpt_backbone.py               ← ONNX inference wrapper
  figures/
    fig_*.pdf                         ← 10 paper figures
    captions.md
  notebooks/
    reproduce_paper.ipynb             ← end-to-end notebook
  docs/
    reproducibility.md                ← detailed walkthrough
    corpus_setup.md                   ← TUH download
    seeds.md                          ← every RNG used
    zenodo_setup.md                   ← DOI workflow
```

---

## Licence

MIT — see `LICENSE`.

The bundled per-recording predictions reference TUH recordings by their
NEDC-issued alphanumeric identifiers (e.g. `aaaaabdo_s003_t000`); these
identifiers are released by NEDC under the same data-use agreement as
the EDFs themselves and contain no patient-identifying information.

## Acknowledgments

- Temple University NEDC group for the TUH-EEG corpus.
- The braindecode / EEGPT authors for the pretrained backbone.
- The IEEE SPMB Symposium reviewers.
