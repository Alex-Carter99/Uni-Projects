# Training configurations for real-time VTaC

Three deep-learning training setups for the real-time ventricular tachycardia (VT) alarm task. They are ranked for a single notebook that should try to beat the published fully convolutional network (FCN). No model was trained for this note, and no extra data were downloaded.

The bar to beat is the real-time FCN in Lehman et al., NeurIPS 2023 Datasets and Benchmarks ([paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7a53bf4e02022aad32a4019d41b3b476-Paper-Datasets_and_Benchmarks.pdf), [code](https://github.com/ML-Health/VTaC)): mean of the five best validation seeds, true-positive rate (TPR) 0.920±0.025, true-negative rate (TNR) 0.855±0.018, PhysioNet Challenge Score 80.08±2.46, area under the ROC curve (AUC) 0.949±0.006. Their best single seed in Table 7 scored 81.96 (TPR 0.949, TNR 0.835, AUC 0.948). Local preprocessing and split details are in `vtac/METHODS.md`.

## What “better” means

Positive class: true VT alarm. The official score, from the PhysioNet/CinC 2015 challenge ([Clifford et al., challenge page](https://physionet.org/content/challenge-2015/1.0.0/)), is

```text
score = 100 * (TP + TN) / (TP + TN + FP + 5 * FN)
```

A missed true alarm costs five times a false alarm kept on. Accuracy is the wrong selection metric: about 71.4% of alarms are false (3,596/5,037; train 2,897/4,060). A constant “false alarm” predictor has accuracy about 0.71, TPR 0, and challenge score 0.

Use only the real-time input the published models use: four channels (ECG, ECG, PPG/PLETH, ABP), 250 Hz, the 10 seconds immediately before onset, shape `(4, 2500)` in channel-first PyTorch layout. Keras InceptionTime uses `(2500, 4)`. Missing channels stay zero. Each present channel is z-scored on that 10-second window; an all-zero channel stays zero. Official patient-level split: train 4,060, validation 495, test 482 (train positives 1,163). Fit every choice on train, select on validation challenge score, and touch the test set once.

At the published operating point (TPR 0.920, TNR 0.855) on the 482-alarm test set (137 true), one extra false negative changes the score by about 0.8 points. The published seed standard deviation is 2.46. A claimed win needs a gap larger than a couple of alarms, reported with the seed spread.

## Shared decision rule

For every configuration:

1. Train with a weighted binary loss so the positive class is not ignored. Compare losses only within the same `pos_weight`. Heavier weights change the loss scale, so they cannot be ranked by training loss.
2. After every epoch, score the validation set. Sweep the threshold on `P(true alarm)` from 0.01 to 0.99 in steps of 0.01. Checkpoint the weights that maximize the validation challenge score. Freeze that threshold for test.
3. Do not use 0.5, and do not use accuracy, to pick the epoch, the weight, the learning rate, or the threshold.
4. If a model’s probabilities were calibrated and a false negative cost exactly five times a false positive, Elkan’s cost-sensitive threshold would be `1 / (1 + 5) = 1/6 ≈ 0.167` (Elkan, “The Foundations of Cost-Sensitive Learning,” IJCAI 2001, [pdf](https://cseweb.ucsd.edu/~elkan/rescale.pdf)). Weighted training moves the probabilities, so 0.167 is only a region where the sweep should be fine. The validation sweep is the rule. Elkan’s own conclusion for probability models is to keep the training prevalence and apply the cost at decision time, which is what the sweep does.
5. Report two test numbers when several seeds are trained: the mean and standard deviation of per-seed scores (comparable to VTaC Table 4) and the score of the averaged probabilities with one threshold chosen on the averaged validation probabilities (the system to deploy). Use all trained seeds. The paper’s “best 5 of 10” filter is optimistic; do not add a second filter on top of an ensemble.
6. Keep augmentations off unless a setting below explicitly allows light Gaussian noise. Do not shift the window onto or past onset (sample 75000). That would break the real-time rule.

Positive-weight grid used below, unless a config narrows it: `{2.5, 3.5, 4.0, 5.0}`. On the training split, `n_neg / n_pos = 2897/1163 ≈ 2.49` and `N / n_pos = 4060/1163 ≈ 3.49`. The paper already searched 3.54 and 4. The value 5 matches the false-negative multiplier in the score. Pick the weight by validation challenge score after the threshold sweep.

## Configuration 1 — InceptionTime ensemble (default)

This is the setup to run first.

**Backbone.** One Inception network as specified by Ismail Fawaz et al., Data Mining and Knowledge Discovery, 2020 ([DOI 10.1007/s10618-020-00710-y](https://doi.org/10.1007/s10618-020-00710-y), [arXiv:1909.04939](https://arxiv.org/abs/1909.04939), [reference code](https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py)):

- Six inception modules. Each module has a bottleneck convolution of width 32 (kernel length 1, used when the input has more than one channel), then three temporal convolutions with lengths 40, 20, and 10, 32 filters each, stride 1, same padding, plus a max-pool of length 3 followed by a kernel-1 convolution of 32 filters. Concatenate the four branches (128 channels), batch-normalize, ReLU.
- Residual addition every third module (modules 3 and 6), with a kernel-1 projection and batch norm on the shortcut.
- Global average pooling, then a single linear logit. Glorot uniform initialization, as in the paper.
- Input `(4, 2500)` or `(2500, 4)` depending on the framework. The convolution runs along time and mixes the four channels in the first layer, which is the right layout for two ECG leads plus PPG and ABP.

The published classifier is an ensemble of five of these networks, same architecture, independent random initializations, equal weight on the probabilities. On the UCR archive a single Inception network had high variance, in the same way their ResNet did, and ensembles larger than five did not help further. VTaC’s FCN score standard deviation of 2.46 is the same kind of problem on a training set of only 4,060 windows.

**Why this is the default against the VTaC FCN.** The VTaC FCN is a widened descendant of the Wang et al. fully convolutional baseline (arXiv:1611.06455; three conv blocks, global pooling). InceptionTime was built to beat that family. Against an ensemble of five Wang ResNets it won 54 datasets, tied 8, and lost 23 (p < 0.01), with batch size fixed at 64 so the gain is not an optimization accident. On 12-lead ECG, Borra et al., Computing in Cardiology 2020 ([DOI 10.22489/CinC.2020.349](https://doi.org/10.22489/CinC.2020.349), [pdf](https://www.cinc.org/archives/2020/pdf/CinC2020-349.pdf)), found a single InceptionTime ahead of ResNet and of xresnet1d depths 18, 34, 50, and 101 on their training cross-validation challenge metric: 0.5183±0.0016 versus 0.5091±0.0071 for ResNet and 0.5180±0.0042 for xresnet1d50. Those numbers are cross-validation on the challenge training data, not the hidden official test set (the authors say they did not receive test scores). InceptionTime had 510,491 parameters in that reimplementation, against 1,808,795 for xresnet1d50 and 3,645,339 for xresnet1d101. About half a million parameters is a sane size for 4,060 windows. They also note the smaller net is the practical one when a condition is rare (premature ventricular contractions had 188 examples in the subset they counted).

Keep the default kernel set `{10, 20, 40}`. The same paper found that longer sets, up to `{32, 64, 128}`, slightly lowered accuracy on the real UCR collection, which they attribute to overfitting small training sets, even though longer kernels helped on long synthetic series. VTaC windows are long (2,500 samples) and the training set is small, so the UCR result is the one to follow. Global average pooling still reads the whole 10 seconds; ventricular tachycardia is a repeated wide-complex pattern, so a local detector can fire many times in the window.

**Loss.** Binary cross-entropy with logits. `pos_weight` in `{2.5, 3.5, 4.0, 5.0}`, chosen on the validation challenge score. One logit, not a two-class softmax. The reference repository compiles categorical cross-entropy and accuracy; that pairing maximizes a different objective and should be replaced.

**Optimizer and schedule.** Adam, learning rate `1e-3`, betas `(0.9, 0.999)`, weight decay 0. That is `keras.optimizers.Adam()` in the reference repository. Batch size 64. The reference trains for 1,500 epochs and drops the learning rate when training loss plateaus (factor 0.5, patience 50, floor `1e-4`), and it checkpoints training loss. Here a patient-level validation set exists, so change the monitor:

- Reduce the learning rate by 0.5 when the validation challenge score has not improved for 10 epochs, floor `1e-5`.
- Early-stop when the validation challenge score has not improved for 25 epochs.
- Upper bound 150 epochs. The 1,500-epoch training-loss schedule was for UCR sets that often have no held-out split.

Select `pos_weight` with one seed, then train five seeds at that weight. Average the five sigmoid probabilities, then choose a single validation threshold on the average.

**Regularization.** The ensemble, batch norm, early stopping, and weight decay 0 are the published controls. No waveform synthesis. Optional sensitivity, only if training challenge score saturates near 100 while validation score stalls: Adam weight decay `1e-4`, decided by the validation challenge score. Do not add dropout inside the inception module; the reference module has none.

**Evidence.** Ismail Fawaz et al. 2020 as cited above, including the ensemble-size and kernel-length studies. Wang, Yan, and Oates, arXiv:1611.06455, for the FCN/ResNet baseline InceptionTime replaces. Borra et al. 2020 for the 12-lead ECG comparison and the parameter counts. VTaC Table 4 for the number to beat and for the failure of the paper’s own contrastive and generative add-ons.

**Main risk.** The longest kernel is 160 ms and the stack is stride 1, so the receptive field is on the order of a second, not the full 10 seconds. If a false alarm is only obvious from several seconds of pulse absence on ABP or PPG, this network underuses the window. The UCR win was mostly accuracy on univariate archives, not the 2015 challenge score, and Borra’s ECG margin over xresnet1d50 was 0.0003 on training cross-validation of a different metric. Five models times a small `pos_weight` grid is the expensive part of the notebook; the grid should be finished on one seed before the other four are trained.

## Configuration 2 — xresnet1d18, ECG benchmark recipe

Run this second. It is the residual network that has actually been tuned on clinical ECGs, at a depth small enough for 4,060 windows.

**Backbone.** `xresnet1d18` from the Strodthoff, Wagner, Schaeffter, and Samek PTB-XL benchmark (IEEE Journal of Biomedical and Health Informatics, 2021; [DOI 10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989), [arXiv:2004.13701](https://arxiv.org/abs/2004.13701), [code](https://github.com/helme/ecg_ptbxl_benchmarking)). The 1D network is the He et al. “Bag of Tricks” xresnet (CVPR 2019) moved to time. Call it with `input_channels=4`.

The published head, Appendix A of that paper: concatenate global average pooling and global max pooling, then a fully connected layer of 128 units, batch norm, dropout 0.25, ReLU, dropout 0.5, and a linear logit. Their code defaults are `lin_ftrs_head=[128]` and `ps_head=0.5` in `code/models/fastai_model.py`; follow the paper’s 0.25-then-0.5 dropout rather than a single 0.5 if the constructor allows both.

Kernel length: the code default `kernel_size=5` was used with 100 Hz ECGs, where five samples is 50 ms. At 250 Hz, five samples is 20 ms, shorter than a QRS. Default for this task is kernel length 13 (52 ms) on whatever convolutions the constructor binds to `kernel_size`. Run kernel length 5 as the second cell, and keep the length with the better validation challenge score. Do not invent a third length. Strodthoff et al. multiplied filter lengths by five when they moved from 100 Hz / 500 samples to 500 Hz / 2,500 samples (Appendix B, their Table VI) and saw no meaningful gain in diagnostic AUC, so a wider kernel is a hypothesis, not a guaranteed upgrade. xresnet stages downsample, which is the point of this configuration: the receptive field covers a multi-second rhythm, complementing configuration 1.

Use depth 18, not 101. On PTB-XL (about 21,837 records, eight training folds) xresnet1d101 tied inception1d at macro AUC 0.925±0.008 on all statements and had the highest mean on rhythm (0.957±0.015 versus inception1d 0.953±0.011 and `fcn_wang` 0.931±0.009). Those ± values are bootstrap half-widths on one test set, as defined in their Appendix A, and several intervals overlap. Borra et al. 2020, on a multicenter 12-lead training cross-validation, found xresnet1d50 essentially tied with InceptionTime (0.5180±0.0042 vs 0.5183±0.0016) and xresnet1d101 slightly lower (0.5140±0.0047) at 3,645,339 parameters, versus 697,499 for xresnet1d18. If the depth-18 model shows a small train/validation challenge-score gap, depth 50 is the upgrade Borra’s table supports. Depth 101 is the wrong next step on this sample size.

**Loss.** The same weighted binary cross-entropy as configuration 1, `pos_weight` in `{2.5, 3.5, 4.0, 5.0}`. Strodthoff et al. used unweighted binary cross-entropy because PTB-XL is multi-label and they selected on macro AUC. That loss does not encode a five-times false-negative penalty.

**Optimizer and schedule.** AdamW (Loshchilov and Hutter, ICLR 2019) with decoupled weight decay `1e-2`, the fastai learner default in their code, decay applied to weights and not to norm or bias parameters. One-cycle schedule (Smith, arXiv:1803.09820) for 50 epochs, which is their code default (`epochs=50`). Their code default max learning rate is `1e-2` and the default batch size is 128, on shorter 100 Hz windows. Here use batch size 64, and choose the one-cycle max learning rate from `{1e-3, 1e-2}` by validation challenge score. Run all 50 epochs so the cycle completes, and reload the epoch with the best validation challenge score. Do not early-stop in the middle of the cycle; that changes the annealing the recipe depends on.

Train one seed through the small grid (two learning rates, two kernel lengths, four positive weights is more than a first pass needs). Practical order: fix kernel length 13, compare the two learning rates and the four weights, then check kernel length 5 only at the winning learning rate and weight. Then train five seeds at the winning cell and average their probabilities before the threshold sweep.

**Regularization.** Weight decay `1e-2`, dropout 0.25 and 0.5 in the head, and the checkpoint. No time warp, no mixup, no lead reversal. Borra et al. zeroed each lead with probability 0.5 because some 12-lead records contain flat leads. Do not copy that here. Both ECG channels are present for every VTaC event, and VT is an ECG diagnosis; zeroing an ECG lead half the time throws away the class evidence. ABP is already zero in about 64% of events.

Optional noise, off by default: add Gaussian noise with standard deviation 0.05 to z-scored channels, with probability 0.5, and keep it only if the validation challenge score improves. Rahman, Rivolta, Badilini, and Sassi (Sensors, 2023, [23(11):5237](https://www.mdpi.com/1424-8220/23/11/5237)) review ECG augmentation and advise against time inversion, spatial inversion, permutation, and lead shuffling. They also note that scaling a normal QRS can imitate hypertrophy and that stretching the PR interval can imitate atrioventricular block, so a “generic” edit can change the diagnostic class.

**Evidence.** Strodthoff et al. 2021, Table II and Appendix A–B, and the training defaults in `code/models/fastai_model.py` and `code/configs/fastai_configs.py`. He et al., CVPR 2019, for the xresnet tricks. Borra et al. 2020 for depth versus parameter count on 12-lead ECG. Ismail Fawaz et al., Data Mining and Knowledge Discovery, 2019 ([DOI 10.1007/s10618-019-00619-1](https://doi.org/10.1007/s10618-019-00619-1)), reported that the plain Wang ResNet was the strongest deep model on the UCR archive overall and that FCN beat it on the ECG datasets in that archive. That is why this configuration is an xresnet with an ECG head, not a drop-in of the 2017 Wang ResNet.

**Main risk.** PTB-XL gains over `fcn_wang` are small and the bootstrap intervals overlap, and that task is multi-label statement detection rather than a 5× false-negative alarm score. Strided residual stages can blur QRS morphology. Depth 18 can still overfit 4,060 windows if the learning rate stays at `1e-2`; the learning-rate pair exists for that reason. A higher TNR with a lower TPR will look like an ECG-benchmark win and still lose on the challenge score.

## Configuration 3 — published FCN, five-seed probability average

Run this third, in the same notebook, as the control that stays on the architecture that already won this dataset.

**Backbone.** The VTaC real-time FCN, unchanged. From the paper and the official training code, as recorded in `vtac/METHODS.md`: three convolutional layers, 4→128 with kernel 51, 128→256 with kernel 25, 256→128 with kernel 13, batch norm, dropout, adaptive max pool, then a dense head. Input `(4, 2500)`. This is the Wang FCN idea with kernels widened for 250 Hz. It is the only network with a published challenge score of 80.08±2.46 on this split.

The underlined winning cell of their hyperparameter table did not survive text extraction, so it is not restated here. Search only inside the ranges they published, narrowed so the notebook stays small: batch size 64, learning rate in `{1e-4, 1e-3}`, dropout in `{0.1, 0.3}`, `pos_weight` in `{3.54, 4, 5}`. The first two weights are the paper’s search set. The extra 5 is the false-negative multiplier and is the only new loss value.

**Loss.** Binary cross-entropy with logits and the positive weight above. Keep this loss. Focal loss, LDAM, and asymmetric loss are discussed below and are not part of this configuration.

**Optimizer and schedule.** Adam, weight decay 0.005, as in the paper. Upper bound 200 epochs. Each epoch, sweep the validation threshold and checkpoint the challenge score. Stop after 40 epochs without an improvement. If the score is still improving at epoch 200, continue toward the paper’s cap of 500. Their write-up says “up to 500 epochs”; checkpointing on the challenge score replaces “train a long time and hope the last epoch is the right one,” which their text does not fully specify.

**Threshold and ensemble.** Train five seeds at the chosen cell. Deploy the average of the five sigmoid probabilities, with one threshold fit on the averaged validation probabilities. Also report the mean and standard deviation of the five separate test scores so the result can be compared with Table 4. This is a deep ensemble in the sense of Lakshminarayanan, Pritzel, and Blundell, NeurIPS 2017 ([“Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles”](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)): independently initialized networks, predictions averaged. InceptionTime uses the same averaging for a different backbone. VTaC published the mean of per-model scores, not the score of the averaged predictor.

**Regularization.** Dropout inside the published search set, Adam weight decay 0.005, and the checkpoint. No waveform augmentation. The published FCN did not need any.

**Evidence.** VTaC Table 4 is the direct evidence that this backbone is the strongest real-time model they tried (FCN 80.08, CNN 76.17, CNN+CL 79.07, FCN+CL 78.41, transformer 62.73). Ismail Fawaz et al. 2020 use a five-model average specifically because single deep TSC models are unstable on small training sets. The published FCN seed spread (score 80.08±2.46) is large enough that averaging is a real, different estimator, not a cosmetic change.

**Main risk.** The ensemble can land inside the published ±2.46 band and fail to clear 80.08. Averaging helps when seeds disagree; if the five networks make the same mistakes on the hard true alarms, the average will not recover those false negatives. The test set has 137 true alarms, so the score moves about 0.8 points per extra false negative. This configuration does not test a new inductive bias. If it matches the paper and configurations 1 and 2 do worse, the notebook has shown that the FCN is already capacity-matched to this split.

## Why the usual imbalance toolkit is not the first attempt

### SMOTE on waveforms

SMOTE (Chawla et al., Journal of Artificial Intelligence Research, 2002) builds synthetic minority points by interpolating a tabular feature vector with a nearest neighbor. On a `(4, 2500)` waveform that interpolation mixes QRS complexes from different patients and different leads, and it mixes real signals with the zeros used for missing ABP or PPG. The result is not a VT rhythm.

van den Goorbergh, van Smeden, Timmerman, and Van Calster (Journal of the American Medical Informatics Association, 2022, [DOI 10.1093/jamia/ocac093](https://doi.org/10.1093/jamia/ocac093)) compared no correction, random undersampling, random oversampling, and SMOTE for clinical risk models. Resampling did not raise AUROC relative to an uncorrected model. It did overestimate minority-class probabilities, SMOTE most severely, and a threshold shift produced the same kind of sensitivity/specificity tradeoff without synthetic samples. Elkan (2001) already argued that, for models which output probabilities, the cost belongs in the decision threshold. The VTaC prevalence is only about 2.5:1 on the training split, not a rare-event regime where synthetic samples are the only way to see a positive.

### Accuracy as the selection metric

Accuracy treats a false negative like a false positive. The 2015 score does not. Selecting the epoch, the threshold, or the checkpoint by accuracy favors the 71% false-alarm class and can raise TNR while the challenge score falls. Gao, Zhang, Lu, and Wang (Journal of Healthcare Engineering, 2019, [DOI 10.1155/2019/6320651](https://doi.org/10.1155/2019/6320651)) report an LSTM with focal loss at 99.26% accuracy versus 98.70% for cross-entropy on MIT-BIH beats. That paper chose hyperparameters by test-set accuracy, including the focal-loss focusing parameter. It is evidence about a different metric on beat snippets, not a reason to select VTaC models by accuracy.

### The paper’s contrastive and generative stack

On this dataset those add-ons already lost to plain FCN. Real-time means of the five best validation seeds (VTaC Table 4): FCN+CL challenge score 78.41±0.87, CNN+CL 79.07±0.99, BeatGAN 41.02±2.71, TanoGAN 47.61±0.87, Diffusion+CL 52.51±2.61, against FCN 80.08±2.46. Contrastive learning raised the CNN (76.17 to 79.07) and lowered the FCN. The authors report that generative models which had looked strong on MIMIC-derived false-alarm data underperform here. With 4,060 labeled windows, a second contrastive view or a diffusion model adds variance before the backbone and the decision threshold have been changed. Those stacks are a later experiment, after configurations 1–3 have scores.

### Focal loss, LDAM, asymmetric loss, class-balanced loss

Focal loss (Lin et al., ICCV 2017, [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)) down-weights easy examples. It was built for dense detection, where each image produces 10^4–10^5 candidate locations and only a few are positive. RetinaNet’s usual settings are a focusing parameter of 2 and a positive balance of 0.25. VTaC training prevalence is about 28.6%. A focusing parameter of 2 also down-weights easy true alarms, which are exactly the cases the 5× term needs the model to keep. It is reasonable only as a later ablation with focusing parameter in `{0, 1}`, and only if the validation challenge score beats weighted cross-entropy at the same positive weight. Gao et al. 2019, cited above, is not that evidence: the imbalance there is beat-level and the reported win is accuracy.

LDAM (Cao et al., NeurIPS 2019, [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)) enlarges the margin of small classes, with the margin scaling like `n^(-1/4)`, and defers reweighting. The published gains are on long-tailed vision (CIFAR with ratios up to 100:1, iNaturalist 2018). On this training split the margin ratio between 1,163 and 2,897 positives and negatives is only about 1.26. LDAM targets balanced error, not a 5× false-negative cost. Deferred reweighting is a sensible trick when reweighting destabilizes training; it is not the first lever at this prevalence.

Asymmetric loss (Ridnik et al., ICCV 2021, [pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)) is a multi-label loss. Its published operating point (negative focusing 4, positive focusing 0, probability clip 0.05) assumes few positive labels per image, as on MS-COCO and Open Images. One binary label at 29% positive is outside that regime. Clipping easy negatives can discard false-alarm gradients and drop TNR.

Class-balanced loss (Cui et al., CVPR 2019, [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)) reweights by the effective number of samples `(1-β^n)/(1-β)`. As β approaches 1 this approaches inverse frequency. At β = 0.999 the positive/negative weight ratio on this training split is about 1.4. At β = 0.9999 it is about 2.3. Both sit at or below the inverse-frequency weight (about 2.5) and below the 3.54 and 4 the paper already searched. The positive-weight grid in configurations 1–3 already covers that range and the cost multiplier. A separate class-balanced loss is redundant as a first run.

### Transformers and large generative models

The VTaC transformer (one encoder layer, model dimension 128, eight heads) scored 62.73±2.78 real-time, against the FCN at 80.08. Strodthoff et al. found convolutional models ahead of LSTMs on PTB-XL (LSTM macro AUC 0.907±0.010 on all statements, versus 0.925±0.008 for xresnet1d101 and inception1d). A from-scratch time-series transformer, or a language-scale pretrained model, is a poor default on 4,060 windows. PatchTST-style models and large generative pipelines are forecasting or pretraining tools; they are not the first classifier for this score.

### Augmentations that damage the label

Do not time-reverse the window. Stabenau, Bridge, and Waks (Computers in Biology and Medicine, 2021, [DOI 10.1016/j.compbiomed.2021.104408](https://doi.org/10.1016/j.compbiomed.2021.104408)) point out that a temporally reversed ECG puts the T wave before the QRS and can reinforce features that do not occur at test time. Do not time-warp or permute segments across the window: VT is defined by consecutive ventricular beats and their rate, so stretching the R–R intervals changes the label. Rahman et al. (2023), cited under configuration 2, likewise advise against time inversion and permutation. Do not crop into the post-onset minute. Do not mix raw windows across patients. Light Gaussian noise at the scale in configuration 2 is the only augmentation these three setups allow, and it is off unless validation challenge score rises.

## Order in the notebook

1. Configuration 1, one seed, positive-weight grid, then five seeds and the probability average.
2. Configuration 2, the reduced grid (kernel, learning rate, positive weight), then five seeds.
3. Configuration 3, the reduced FCN grid, then five seeds and the probability average.
4. Compare on validation challenge score first. Evaluate test once, for the validation winner of each configuration, and report TPR, TNR, and the challenge score at the frozen threshold, plus AUC.

None of these configurations has a published VTaC challenge score. Beating 80.08 is a hypothesis supported by results on other waveform tasks and by the seed variance of the published FCN, not a number already measured on this test set.
