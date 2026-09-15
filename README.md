# esf-recoverability — v2 (preprint v2, September 2026)

> **v2 supersedes v1.** v1 (30 June 2026) was submitted to IEEE SPMB 2026 and withdrawn by the author after
> finding that its acquisition operators were applied *after* canonicalization and that its probe had been trained on a
> split overlapping the evaluation set. v2 re-implements the sweep upstream of the contract, trains every probe on the
> TUAB **train** split only, and extends the study to three frozen backbones (EEGPT, LaBraM, BIOT) with four probes each.
> Nothing from the v1 run is used in v2; the v1 snapshot remains at tag `v1.0.0-spmb2026-submission` for the record.

**Paper:** *ESF: Harmonizing Heterogeneous EEG Acquisition for Pretrained Classifiers* — preprint v2, Zenodo DOI 10.5281/zenodo.22772008. Code archive: this release, tag `v2.0.0-preprint-v2`, DOI 10.5281/zenodo.22771925.

## What is in `v2/`

| Path | What it is |
| --- | --- |
| `v2/esf_v2/pipeline.py` | The ESF contract (C1–C6) as a single `run(...)` with per-stage switches; `canonical()` = full contract |
| `v2/esf_v2/acquisition.py` | The five acquisition operators (native rate, gain, mains, montage, broadband SNR) + the low-pass measurement axis; all act on raw µV at native rate |
| `v2/esf_v2/backbones.py` | Frozen-backbone adapters: EEGPT (250 Hz, 4 s), LaBraM (200 Hz, 15 s), BIOT (200 Hz, 5 s, 16 TCP bipolar) |
| `v2/extract_features.py`, `v2/train_head.py` | Per-recording features on TUAB train/eval; MLP (seeds 7/11/13) and linear probes, Platt on validation |
| `v2/extract_cells.py`, `v2/score_cells.py` | The 46-cell sweep: per-cell feature cache (float16) scored by every probe without a GPU |
| `v2/run_permutation_control.sh`, `v2/outly.py` | Permuted-label probes; unsupervised outlyingness (kNN, whitened norm, random heads) |
| `v2/pinsker_v2.py`, `v2/monitor_v2.py` | Class-conditional AUROC–TV bound check; label-free monitor test |
| `v2/bench_contract.py` | Per-stage CPU cost of the contract and CPU cost of each backbone |
| `v2/legacy_vignette.py`, `v2/vendor/natus_e_parser.py` | The 125 Hz Natus `.e` vignette (recordings not released) |
| `v2/aggregate_v2.py`, `v2/tables_v2.py`, `v2/table_failure_multi.py`, `v2/fig_story2.py` | Every table and figure in the paper from `v2/results/` |
| `v2/results/` | `sweep_<backbone>_<probe>_s<seed>.csv`, `per_recording_predictions_*.csv` (every cell, every probe), permutation controls, Pinsker summaries, `bench_a10.json`, `legacy.json` |
| `v2/head_<backbone>_<probe>_s<seed>.pt` | All twelve probes with scalers and Platt parameters |

Feature and cell caches (~12 GB) are not archived; `v2/chain_backbone.sh <backbone>` regenerates them from TUAB in a few GPU-hours. Every table and figure regenerates from the released per-recording predictions on a CPU:

```bash
python v2/aggregate_v2.py v2/results/ v2/agg/ && cd v2/agg && python ../tables_v2.py . ../results && cd .. && python fig_story2.py agg results ../data paper
```

Data provenance, checkpoint revisions and seeds are listed in the paper's Appendix E. TUAB v3.0.1 is obtained under the NEDC data-use agreement and is not redistributed.

## Citation

```bibtex
@misc{dammu2026esf,
  author    = {Hitesh Dammu},
  title     = {{ESF}: Harmonizing Heterogeneous {EEG} Acquisition for Pretrained Classifiers},
  year      = {2026},
  note      = {Preprint v2, Zenodo},
  doi       = {10.5281/zenodo.22772008},
  url       = {https://doi.org/10.5281/zenodo.22772008},
}

@software{dammu2026_esf_recoverability,
  author    = {Hitesh Dammu},
  title     = {esf-recoverability: reference implementation, probes and per-recording results},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v2.0.0-preprint-v2},
  doi       = {10.5281/zenodo.22771925},
  url       = {https://doi.org/10.5281/zenodo.22771925},
}
```

## Quickstart (CPU only)

```bash
git clone https://github.com/h5371h/esf-recoverability.git && cd esf-recoverability
python3 -m venv .venv && source .venv/bin/activate && pip install -r v2/requirements.txt
python v2/aggregate_v2.py v2/results/ v2/agg/
cd v2/agg && python ../tables_v2.py . ../results && cd ..
python fig_story2.py agg results ../data paper
```

This regenerates every table and figure of preprint v2 from the shipped per-recording predictions. To regenerate the predictions themselves you need TUAB v3.0.1 (NEDC data-use agreement) and a GPU: `bash v2/chain_backbone.sh eegpt|labram|biot` runs feature extraction, the four probes, the 46-cell cache and scoring for one backbone.

## v1 (withdrawn)

The first version of this work (30 June 2026) was submitted to IEEE SPMB 2026 and withdrawn by the author; its code, data and figures remain at tag `v1.0.0-spmb2026-submission` (Zenodo 10.5281/zenodo.22727770) for the record and should not be used. The `src/`, `data/*_final.csv`, `figures/`, `notebooks/` and `docs/` trees from v1 are kept for that reason; the v1 failure-taxonomy script in `src/` is reused by v2 with v2 inputs.

## Licence

MIT, see `LICENSE`. Per-recording predictions reference TUH recordings by their NEDC identifiers, which carry no patient-identifying information. No clinical recordings are included.

## Acknowledgments

Temple University NEDC for the TUH EEG corpus; the EEGPT, LaBraM and BIOT authors and the braindecode maintainers for open weights and loaders.
