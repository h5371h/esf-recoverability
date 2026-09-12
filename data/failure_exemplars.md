# Failure-Taxonomy Exemplars

Median (typical) recording per category. `pred (prob) ✓/✗` reads as the predicted class, the abnormal-class probability, and a check/cross for correctness against ground truth. Probes pulled from the canonicalized arm at the most aggressive severity per axis (sampling_rate=128 Hz, calibration gain=0.5×, power_line=30 µV).

| rec_id | gt | baseline pred | sr=128 (canon) | cal=0.5 (canon) | pl=30 (canon) | category | flip_rate | max wrong-side conf |
|--------|----|---------------|----------------|-----------------|----------------|----------|-----------|---------------------|
| `aaaaamsc_s001_t000` | abnormal | abnormal (0.99) ✓ | abnormal (1.00) ✓ | abnormal (1.00) ✓ | abnormal (0.99) ✓ | **Robust-correct** | 0.00 | 0.00 |
| `aaaaagyr_s002_t000` | abnormal | normal (0.48) ✗ | abnormal (0.96) ✓ | abnormal (0.93) ✓ | abnormal (0.50) ✓ | **Shift-recovered** | 0.82 | 0.64 |
| `aaaaalsx_s001_t000` | abnormal | abnormal (0.62) ✓ | abnormal (0.83) ✓ | abnormal (0.97) ✓ | abnormal (0.65) ✓ | **Axis-specific** | 0.18 | 0.68 |
| `aaaaajna_s001_t000` | normal | normal (0.17) ✓ | abnormal (0.67) ✗ | abnormal (0.61) ✗ | normal (0.19) ✓ | **Shift-induced flip** | 0.36 | 0.73 |
| `aaaaajfe_s001_t000` | normal | normal (0.08) ✓ | abnormal (0.95) ✗ | abnormal (0.81) ✗ | normal (0.08) ✓ | **Catastrophic shift** | 0.36 | 0.98 |
| `aaaaaicn_s001_t000` | normal | abnormal (0.64) ✗ | abnormal (0.92) ✗ | abnormal (0.97) ✗ | abnormal (0.62) ✗ | **Robust-wrong** | 0.18 | 0.98 |
