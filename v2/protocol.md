# ESF acquisition-shift sweep — v2 protocol (corrected)

## Why v2 exists
Inspection of the v1 code (`perturbations.py`, `eval_loop.py`) against the v1 text found:
1. **Injection point.** Every v1 perturbation was applied to the finished ESF signal (already resampled to 250 Hz, notched,
   average-referenced and robust z-scored), although Fig. 1 states the axes inject upstream of canonicalization.
2. **Sampling-rate axis.** The naive arm was an identity (all naive rows equal the baseline). The canonicalized arm passed
   the 250 Hz signal to `resample_poly` as if it were at the native rate, producing a time-warp (spectral scaling by
   f_native/250), not a sampling-rate acquisition. No recording was ever simulated at 128/200/256/512 Hz.
3. **Gain axis.** The canonicalized arm divided the clean signal by g without first multiplying by g; no recovery was tested
   (canon g = naive 1/g to seven decimals for that reason).
4. **Line-noise units.** Amplitudes labelled µV were added to a z-scored signal, i.e. they were in robust-SD units.
5. **Probe training split.** Head A v1 was trained on a patient-stratified split of the *whole* TUAB corpus
   (2,397/303/293 = 2,993 recordings), so most of the 276 evaluation recordings were in the probe's training set.

## v2 design
* Acquisition operators A_theta act on the **raw microvolt EDF at its native rate** (upstream of ESF).
* **Canonicalized arm** = the full ESF pipeline (map → resample to 250 Hz → notch at regional mains + harmonics →
  0.5 Hz HP → common average reference over good channels → per-channel robust z-score).
* **Naive arm** = the same pipeline with the axis's corrective stage disabled:
  sampling rate: no resample (native samples treated as 250 Hz); gain: per-recording robust z replaced by fixed
  per-channel scale constants (median of medians / median of IQRs over 300 training recordings); line noise: notch off;
  montage: CAR over all 19 slots with missing channels as zeros; broadband: no corrective stage (arms identical by design).
* **Operators.** A1 anti-aliased polyphase resample to f_native ∈ {128,200,250,256,512}. A2 raw × g, g ∈ {0.5,0.75,1,1.5,2}.
  A3 additive mains at the regional frequency (60 Hz for TUH) with harmonics below the native Nyquist only, per-electrode
  log-normal amplitude (σ 0.3) and random phase, mean peak amplitude ∈ {0,5,15,30,60} µV. A4 {full19, legacy16 (drop
  T3,T4,T5,T6,Fz,Cz,Pz at the raw level), bipolar18 (longitudinal derivations re-projected to anodes)}.
  A5 per-channel white Gaussian noise at SNR ∈ {−5,0,5,10,20} dB. Seeds: crc32(rec_id|axis|severity).
* **Backbone.** braindecode 1.5.2 `EEGPT`, public `braindecode/eegpt-pretrained` weights, frozen; the 19→19
  `chan_proj` adapter is not in the checkpoint and is initialised with seed 7 and frozen (the v1 ONNX embedded a different
  random draw; the weights file was lost with the storage account). 1000-sample windows at 250 Hz, stride 1000,
  encoder output averaged over patches and embeddings → 512-d, L2-normalised per window.
* **Probe (Head A v2).** Trained on TUAB v3.0.1 **train only** (2,717 recordings), patient-stratified 90/10 train/val
  inside train; recipe unchanged from v1 (Linear(1536,256)-ReLU-Dropout(0.2)-Linear(256,1), AdamW 1e-3/1e-4,
  BCE with pos_weight, 25 epochs, batch 256, best-val-AUC checkpoint, StandardScaler on pooled features); Platt scaling
  fitted on the val split. Recording-level score: mean of up to 8 group logits (v1 contract), variance across groups
  retained as `var_logit`. Evaluation on the official 276-recording eval split, never seen in training.
* Output schema identical to v1 so all downstream statistics and figure scripts run unchanged.
