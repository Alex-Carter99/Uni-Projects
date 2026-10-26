# VTaC methods note

Sources checked: NeurIPS 2023 Datasets and Benchmarks paper (Lehman et al., PDF text extracted with `pdftotext -layout`) and the official code at https://github.com/ML-Health/VTaC (`preprocessing/filtering.py`, `preprocessing/standardize.py`, and the `mlp` / `cnn` / `fcn` / `Transformer` / `sae` train and eval scripts). Dataset: PhysioNet VTaC v1.0, https://physionet.org/content/vtac/1.0/, CC BY-SA 4.0, DOI https://doi.org/10.13026/8td2-g363. The code README lists a different DOI (10.13026/z4f3-1f07); the version DOI above is the one on the v1.0 release used here.

## Task

Predict whether a bedside **ventricular tachycardia (VT) alarm is true or false**.

VT, following the PhysioNet Challenge 2015 definition used in the paper, is five or more consecutive ventricular beats at a heart rate above 100 bpm. The positive class is a **true alarm** (label `True` / code label 1). A false negative is a true VT alarm called false. That error is treated as much more costly than a false positive.

Class balance on the 5,037 events kept for modeling (Table 1): **1,441 true (28.61%)** and **3,596 false (71.39%)**. Uncertain (102) and Reject (211) events are excluded. About 78.96% of the 5,742 annotated events had unanimous two-annotator labels; conflicts that were adjudicated are included, and 392 un-adjudicated conflicts were dropped.

## Data

- Three US hospitals and three commercial bedside-monitor vendors. The paper does not name the hospitals or vendors. Recordings span 2013–2022 and do not overlap the 2015 PhysioNet Challenge. Source pool: 20,158 continuous recordings (mean length 4.5 days) → 18,465 VT alarm events from 2,376 records → at most five alarms per record (5,742 annotated) → 5,037 final true/false events from 2,260 records.
- Native rates varied (ECG at least 240 Hz; ABP/PPG at 100, 125, or 240 Hz). **Everything is resampled to 250 Hz.**
- Every event has at least two ECG leads (median 3, mean 3.4). ABP is present in about 36% of events and PLETH/PPG in about 91% (Table 2: ABP 36%, PLETH 91%; Table 3 is slightly more precise at 35.7% and 90.6%).
- The collection write-up describes a 10-minute clip (5 min before and 5 min after onset). The **released records used by the official code are 90,000 samples (6 minutes)** at 250 Hz: 5 minutes before onset and 1 minute after. Onset is sample **75000**.
- **Real-time:** samples `[72500:75000]`, the 10 seconds immediately before onset (2,500 samples). No samples after onset. This matches the 2015 Challenge “real-time” rule (data only up to onset). The paper’s retrospective window is shorter than the 2015 Challenge’s 30 seconds after onset.
- **Retrospective:** samples `[72500:76250]`, 15 seconds = 10 s before onset plus 5 s after (3,750 samples).
- Each model sees **four channels**: two ECG leads, one PPG, one ABP. The paper does not say which ECG leads are kept. Official `filtering.py` assumes a prebuilt tensor ordered ECG, ECG, PPG, ABP. Missing channels are filled with 0. The lead-selection script that builds that tensor is not in the GitHub repo.
- Split is **by patient record**, not by alarm, about 80/10/10 (Table 3):

| | Overall | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| Alarms | 5,037 | 4,060 | 495 | 482 |
| Records | 2,260 | 1,808 | 226 | 226 |
| True alarms | 1,441 | 1,163 | 141 | 137 |
| % true | 28.61% | 28.65% | 28.90% | 28.48% |
| ECG | 100% | 100% | 100% | 100% |
| ABP | 35.7% | 35.6% | 36.5% | 35.6% |
| PLETH | 90.6% | 90.42% | 89.83% | 92.32% |

Table 3 also lists adjudicated / unanimous counts. Train (542 + 3,518) and validation (46 + 449) add up to the alarm counts. The test cells are printed as 54 adjudicated and 438 unanimous, which sum to 492 rather than 482. Overall 642 + 4,395 = 5,037. Treat the test agreement split as a table inconsistency, not as a second test size.

## Preprocessing

Paper (ECG): (1) high-pass at 1 Hz for baseline wander, (2) second-order Butterworth low-pass at 30 Hz, (3) notch for power-line interference. The code uses 60 Hz notch, Q = 30, which the paper text does not state explicitly.

Paper (PPG): high-pass with stopband 0.3 Hz and passband 0.5 Hz, and low-pass with passband 5 Hz and stopband 8 Hz. The code is simpler: 60 Hz notch, then a first-order Butterworth **bandpass at 0.5–5 Hz**.

ABP is not given filter cutoffs in the paper text. The code and its README use a **60 Hz notch and a 16 Hz low-pass**.

Z-score: the paper normalizes each signal segment by its own mean and standard deviation. `standardize.py` option 2 (the one `train.py` loads from `sample-norm`) computes mean and std on samples 72500:75000 of that channel and applies them to the channel. Channels that are all zeros (missing) are left as 0. NaNs are set to 0 before training.

## Models and training

Benchmarked in the paper:

1. **Rule-based** — Plesinger et al. 2015 (a winning 2015 Challenge entry): QRS / R-R regularity on each channel. Code is the Challenge 2015 sources, not this repo.
2. **MLP** — three hidden layers, 1024 / 256 / 64, then a linear output. Flattened waveform in, BCE-with-logits.
3. **SAE** — supervised autoencoder. Encoder and decoder are four conv layers each; a four-layer MLP reads a 32-d vector. Loss is reconstruction plus cross-entropy. Input is the whole 10 s segment, not beat-by-beat FFT as in Lehman et al. 2018.
4. **Transformer** — one encoder layer, d_model 128, 8 heads, GELU, dropout 0.1. A linear layer maps each time step to 128-d. BCE.
5. **CNN** — four 1-D conv extractors, each two conv layers, kernel widths 50, 100, 200, and 400 in the paper (the released `nets.py` defaults are 25, 50, 100, 150). Features are pooled, concatenated, and passed through a 128-d hidden layer. The repo’s CNN forward path is the Zhou et al. 2022 lightweight 1-D CNN.
6. **CNN+CL** — same CNN plus a contrastive / discriminative term.
7. **FCN** — three conv layers (code: 4→128 k=51, 128→256 k=25, 256→128 k=13), batch norm, dropout, adaptive max pool, then a dense head. This is the best real-time model.
8. **FCN+CL** — FCN plus the same contrastive term.
9. **BeatGAN** — unsupervised GAN anomaly detection (Zhou et al. 2019).
10. **TanoGAN** — GAN anomaly detection for small data (Bashar and Nayak 2020).
11. **Diffusion+CL** — conditional diffusion reconstructs a “true alarm” future trajectory; the MSE distance from the observed waveform is the anomaly score (Wu et al. 2023). Generative models use a shorter window than 10 s. Diffusion details in the appendix: 36 residual layers, 200-step linear schedule β 0.0001→0.02, Adam 1e-4, ABP scaled by 1/10, 32 negatives, λ = 0.5.

Contrastive term in the released supervised+CL training code (Zhou-style discriminative constraint, not a separate SimCLR repo): the network encodes the 10 s pre-alarm window and a random 10 s crop taken from earlier in the record (start index uniform in `[0, 75000 - 5000)`). The extra loss pulls false-alarm pairs together and pushes true-alarm pairs apart in that embedding, added to BCE with a weight searched over {0.5, 1.0, 1.5}.

Supervised training choices (paper + code):

- Loss: **BCE with a positive-class weight** (search set {3.54, 4}) plus, for CL variants, the discriminative constraint. Adam, weight decay 0.005. Up to **500 epochs**.
- Grid search on the real-time task over batch size {32, 64, 128}, learning rate {1e-4, 1e-3}, and dropout (typically {0.0, 0.1, 0.3}; CL models {0.1, 0.3, 0.5}). The paper underlines the winning cell in Table 9; the underline does not survive text extraction, so the chosen cell is not repeated here.
- Best validation setting is retrained with **10 seeds**. Reported mean ± sd is the **5 seeds with the best validation Challenge Score**. Retrospective runs reuse the real-time hyperparameters (no second grid search).
- Operating point: threshold on predicted probability that **maximizes the validation Challenge Score**, then frozen for the test set. TPR, TNR, PPV, and F1 use that threshold. AUC does not.
- Generative models (BeatGAN, TanoGAN) use the authors’ GitHub settings with channel count changed, lr 1e-4, weight decay 5e-4, batch 128, and the same validation-score threshold search. Their code is not in ML-Health/VTaC.

## How they score

TPR = TP / (TP + FN), TNR = TN / (TN + FP), with positive = true alarm.

PhysioNet Challenge 2015 score, as written in the paper:

`score = (TP + TN) / (TP + TN + FP + 5 * FN)`

False negatives are weighted **5×**. The tables and the official `eval.py` report this quantity **times 100** (FCN’s 80.08 means 0.8008 on the fraction above).

## Published results (quoted from the PDF tables)

Real-time test performance, mean ± sd of the 5 best-validation seeds (Table 4):

| Method | TPR | TNR | PPV | F1 | Score | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Rule-based | 0.942 | 0.629 | 0.502 | 0.655 | 67.32 | – |
| MLP | 0.597±0.087 | 0.691±0.09 | 0.441±0.037 | 0.502±0.015 | 45.58±1.10 | 0.706±0.008 |
| SAE | 0.848±0.028 | 0.790±0.032 | 0.617±0.028 | 0.713±0.012 | 68.77±1.11 | 0.896±0.007 |
| Transformer | 0.837±0.039 | 0.707±0.060 | 0.535±0.048 | 0.651±0.030 | 62.73±2.78 | 0.852±0.006 |
| CNN | 0.928±0.006 | 0.782±0.019 | 0.629±0.019 | 0.750±0.013 | 76.17±1.20 | 0.936±0.009 |
| **FCN** | **0.920±0.025** | **0.855±0.018** | **0.717±0.024** | **0.805±0.016** | **80.08±2.46** | **0.949±0.006** |
| CNN+CL | 0.930±0.016 | 0.823±0.012 | 0.676±0.012 | 0.783±0.003 | 79.07±0.99 | 0.943±0.005 |
| FCN+CL | 0.931±0.028 | 0.811±0.046 | 0.666±0.048 | 0.775±0.024 | 78.41±0.87 | 0.932±0.008 |
| BeatGAN | 0.597±0.100 | 0.597±0.091 | 0.373±0.033 | 0.455±0.037 | 41.02±2.71 | 0.597±0.028 |
| TanoGAN | 0.704±0.053 | 0.611±0.076 | 0.421±0.027 | 0.524±0.008 | 47.61±0.87 | 0.657±0.012 |
| Diffusion+CL | 0.853±0.054 | 0.517±0.040 | 0.412±0.013 | 0.555±0.016 | 52.51±2.61 | 0.685±0.017 |

The paper’s own summary: FCN is the top real-time model, Challenge Score **80.08±2.46** and AUC **0.949±0.006**. Contrastive learning lifts CNN from 76.17 to 79.07 and does not lift FCN (78.41). The 2015 rule-based entry scores 67.32 real-time. Section 6 says generative models that looked strong on MIMIC-II under-perform here.

Best single seed by validation Challenge Score, real-time test (Table 7). FCN again: **TPR 0.949, TNR 0.835, PPV 0.695, F1 0.802, Score 81.96, AUC 0.948**.

Retrospective means (Table 8) do not raise the headline. Best score there is **FCN+CL 78.95±1.41** (TPR 0.921±0.013, TNR 0.836±0.011, AUC 0.933±0.007). CNN+CL is 78.85±1.95. Plain FCN drops to 75.84±1.86. Rule-based retrospective score is 65.54 (TPR 0.964, TNR 0.571). The paper says only FCN+CL clearly gained from the extra 5 seconds.

## What “better than the paper” means

Use the **official record-level test split** (482 alarms) and the **real-time** input (samples 72500:75000 only).

The number to beat is the **Challenge Score**, not accuracy and not TNR alone:

`100 * (TP + TN) / (TP + TN + FP + 5 * FN)`

with the threshold chosen to maximize that score on the **validation** set.

- Matched to Table 4 (mean of several seeds): **above 80.08**. FCN’s real-time TPR and TNR at that operating point are **0.920±0.025** and **0.855±0.018**. A higher TNR with a lower score is worse by the metric the paper emphasizes, because missed true alarms are weighted 5×.
- Matched to Table 7 (one selected seed): **above 81.96**, where that FCN seed has TPR 0.949 and TNR 0.835.
- AUC to beat in the same real-time table is **0.949±0.006**, but AUC is not the headline.
- Retrospective comparison is a different task. The best published retrospective score is 78.95±1.41, which is still below real-time FCN.

Hardware in the paper: Tesla V100 32 GB, PyTorch 1.9.0, wfdb 4.1.1. That does not change the metric.

## Checked on the unpacked v1.0 files

The PhysioNet zip is 2,866,704,109 bytes. Unzipped under `vtac/data/`, the archive’s own top-level folder is `vtac-a-benchmark-dataset-of-ventricular-tachycardia-alarms-from-icu-monitors-1.0/`. Inside it: `event_labels.csv` (5,037 rows; 1,441 True, 3,596 False), `benchmark_data_split.csv` (train 4,060, val 495, test 482; same record/event keys as the labels), and `waveforms/` (5,037 `.hea` and 5,037 `.dat`). Every label `event` is a header stem. `wfdb.rdrecord` on `003c13_0115` returns signals II, aVR, V, PLETH, fs 250, length 90,000. A second record (`1d125c_0747`) is I, II, III, V, PLETH, ABP, also fs 250 and length 90,000. That matches the code’s 6-minute record and the sample index 75000 alarm onset.
