# TUH-EEG corpus setup

The paper trains and evaluates on the
[Temple University Hospital EEG corpus](https://www.isip.piconepress.com/projects/tuh_eeg/),
maintained by the Neural Engineering Data Consortium (NEDC) at Temple
University.

## Subcorpora used

| Subcorpus | Version | Role | Recordings used in paper |
| --- | --- | --- | --- |
| TUAB (abnormal) | v3.0.1 | Head A training + the sweep eval split | 276 eval (Tier A) + ~ 2 400 train (Tier B) |
| TUEV (events) | v2.0.1 | (optional) Head C / IED-present sweep | n/a in the SPMB paper |

The paper's headline numbers come from the **TUAB eval split only**;
the train split is needed exclusively if you wish to retrain Head A
from scratch (Tier B reproduction).

## Getting access

1. Submit the NEDC data-use form:
   <https://isip.piconepress.com/projects/nedc/html/forms/dat_use.shtml>
2. NEDC replies with an rsync username (typically within 1–2 days).
3. Fetch:
   ```bash
   export TUH_USER=...               # supplied by NEDC
   export TUH_ROOT=$HOME/tuh_corpus  # any path
   mkdir -p "$TUH_ROOT/edf/eval"

   rsync -auxvL \
     "${TUH_USER}@www.isip.piconepress.com:~/data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf/eval/" \
     "$TUH_ROOT/edf/eval/"
   ```
   Approx 8 GB on disk.
4. Cross-check the layout against `data/manifest_tuab_eval.json`:
   every `rec_id` in the manifest should resolve to
   `$TUH_ROOT/edf/eval/{normal,abnormal}/<rec_id>.edf`.

## File layout expected by `eval_loop.py`

```
$TUH_ROOT/edf/eval/
    normal/
        aaaaabdo_s003_t000.edf
        ...
    abnormal/
        aaaaabsk_s007_t000.edf
        ...
```

The label is inferred from the parent folder name (`normal` → 0,
`abnormal` → 1). This matches the TUAB v3.0.1 release layout
verbatim — no rearrangement needed.

## ESF canonicalization

Every recording is converted to ESF (the canonical 19-channel, 250 Hz,
µV, common-average reference representation) before it enters the
EEGPT backbone. This is what the paper calls "canonicalization". The
implementation lives in `src/esf/` and is what the
`canonicalized` arm of every perturbation actually runs.

## Citing TUH

If your downstream work uses the TUH corpus, please cite:

> López, S., Yang, F., Wernecke, M., et al. "The Temple University
> Hospital EEG Corpus: Electrode Location and Channel Labels."
> *Frontiers in Neuroscience* 9 (2015): 196.

And:

> Obeid, I., Picone, J. "The Temple University Hospital EEG Data Corpus."
> *Frontiers in Neuroscience* 10 (2016): 196.
