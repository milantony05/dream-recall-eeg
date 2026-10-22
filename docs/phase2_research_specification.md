# Phase 2 — research specification (pre-results)

Version: 1.0, 22 October 2026. This specification follows [Phase 1](phase1_feasibility_audit.md). It commits to a modest course-project question while leaving raw-signal and label-integrity gates explicit. **No model has been trained and no result is implied.** Any post-audit change to the primary cohort, outcome, window or evaluation must be documented with its reason before looking at test results.

## 1. Title, scope and problem statement

**Final course-project title:** *Participant-Held-Out Prediction of Dream Reports from Pre-Awakening N2 EEG.* The paper title, if warranted, will be revised to reflect actual findings; WR analysis remains exploratory.

At an awakening from N2 (the second non-REM sleep stage), a participant reports whether they recall an experience, have the impression of an experience without recalling content, or report no experience. The primary question is whether an EEG signal recorded **before** the awakening helps predict **recalled experience (E) versus no reported experience (NE)** for a previously unseen participant from the same study. EEG records voltage fluctuations at the scalp. A "white dream" / without-recall report (WR) is an uncertain subjective report category, not evidence that a specific experience objectively occurred.

The aim is to predict the *report after awakening*, not the contents of a dream or the presence/absence of consciousness. The intended audience is researchers studying sleep reports and ML evaluation, rather than a clinical diagnostic or consumer dream-reading product.

### Research questions and decision rules

| ID | Question / hypothesis | Evidence that would support it | Evidence that would not |
|---|---|---|---|
| RQ1, primary | Within N2, does EEG improve E-versus-NE discrimination over a training-fold class-prior baseline for unseen participants? | Participant-held-out discrimination and macro F1 improve with participant-cluster confidence intervals excluding a trivial/no-improvement difference, and the gain survives prespecified confound checks. | Only epoch-random gains, a gain explainable by participant/site metadata, or wide uncertainty including no improvement. |
| RQ2, secondary | Does a compact EEG model improve on band-power logistic regression? | A paired participant-level comparison with uncertainty favors it consistently, without tuning on test folds. | A higher point estimate alone, or a result that vanishes on held-out people. A tie is an informative result. |
| RQ3, exploratory | Can WR be distinguished from E and NE, especially E versus WR? | Direct three-class, held-out-participant WR precision/recall and calibrated probability are meaningfully above a prior-only comparator, with uncertainty reported. | A binary model merely assigning WR to E, rare-class accuracy inflated by prevalence, or one study's result claimed to generalize elsewhere. |
| RQ4, conditional external check | Does a model trained on one study transfer to a separate laboratory? | A genuinely held-out study with compatible signal/label definitions has nontrivial E/NE performance against a source-prior baseline and intervals; three-class transfer requires adequate WR support. | Failure to harmonize, a target lab used for training/calibration, or apparent gains dominated by laboratory prevalence. |

These are testable questions, **not predicted outcomes**. The primary contribution for a course project is transparent, leakage-resistant measurement; a paper claim depends on actual evidence and a refreshed literature review.

## 2. Dataset lock and label mapping

### Primary cohort

Use **Tononi Serial Awakenings amendment 2 / package v3** (DREAM set 13, https://doi.org/10.26180/23306054.v3), N2 only. Catalogue candidates: **287 awakenings / 39 participants: 124 E, 56 WR, 107 NE**. The **primary binary** dataset comprises the 231 E/NE candidates; WR is held out of the primary binary fit and assessment, then used in a separately trained exploratory three-class assessment. These counts are before file checks and artifact exclusions. Version 2's 261 reports must never be mixed with version 3 or counted as additional observations.

A verified `(set_id, amendment, subject_id, case_id/filename)` is the awakening key. The participant grouping key is `(study identity, subject identity)`, with cross-study identity checked before any proposed joint analysis. No participant ID, filename, report text or post-awakening information enters model input.

### External cohort, only after compatibility audit

LODE (set 3, https://doi.org/10.6084/m9.figshare.22147085.v1) has **118 N2 catalogue candidates across 27 N2 participants: 66 E, 11 WR, 41 NE**. Its seven EEG channels and 250 Hz recordings differ substantially from Tononi's high-density 500 Hz setup. It is a possible **locked, held-out-site binary check**, not a guaranteed pooling partner. Compare only features that can be defined consistently *without using LODE labels or outcomes*. Direct CNN transfer is omitted unless a defensible, matching montage/reference can be verified in advance. Its 11 WR samples are too few for a strong independent three-class conclusion.

Other DREAM sets remain outside the primary protocol. Additional studies could be a separately versioned extension after a file-level audit and a refreshed protocol; do not enlarge the evaluation set after seeing scores.

### Inclusion and exclusions

Include an awakening only if its current-amendment catalogue row matches a real EEG file and report, has an unambiguous E/NE/WR classification and participant ID, is marked N2 at the relevant pre-alarm period, and has a valid contiguous pre-awakening window. Verify montage, sampling rate, units, reference, timestamps, and stage alignment from package metadata/EDF and original study description. Document every mismatch and exclusion in a frozen row-level manifest, with counts by label and participant both before and after exclusions.

Exclude W, N1, N3 and REM from the primary N2 cohort; unknown or combined labels; missing/mismatched participant or file IDs; non-EEG/MEG-only signals; periods crossing the alarm or containing too little valid EEG; and predeclared gross corruption. A signal-quality rule must be set without consulting test labels. Report quality-related exclusions by class, because unequal rejection itself could bias performance. Do not equate a "no experience" category from a protocol that never elicited WR with a protocol that explicitly distinguishes WR.

**Window gate:** Inspect the Tononi timing markers and EDF headers before fixing samples. The intended primary input is one fixed 30-second segment ending five seconds before verified alarm onset (relative interval `[-35,-5)` seconds). This removes the most immediate arousal/alarm boundary and keeps one prediction per awakening. If alarm onset cannot be established reliably, stop and revise the protocol *before training*. A separately declared earlier window can test timing sensitivity; do not search many windows and report the winner as if prespecified.

## 3. Signal processing and leakage controls

Within the chosen study, select a small set of EEG channels using **anatomical/electrode metadata only**, with exact channel names/coordinates frozen after the raw montage audit and before viewing class-dependent results. Do not select channels based on test-set importance. Remove EOG, EMG and non-EEG channels from the EEG input. Keep a separate EOG/artifact sensitivity check if those channels exist. Units and channel reference must be verified, then apply a documented, modest band-pass and resample to a common rate (proposed 0.5–35 Hz and 128 Hz, provided the EDF sample rate and anti-aliasing allow it). Filtering can be deterministic per recording; any data-adaptive scaling, artifact threshold, imputation, feature selection or calibration is learned from training participants within each outer fold. Apply the trained transform unchanged to validation/test data.

One pre-awakening segment represents one awakening. If technical subwindows are used for feature calculation, aggregate to a single awakening-level prediction and keep every subwindow from one participant on the same side of every split. Never inflate the test sample count with correlated seconds.

## 4. Splits, baselines and models

**Primary evaluation:** Five outer folds grouped by participant within Tononi N2, using a deterministic seed and a group-aware stratification attempt. Audit the actual E/NE and WR counts per fold; if a fold lacks a primary class, use fewer grouped folds fixed before modeling. The untouched outer fold supplies test predictions once. Within each training portion, use group-aware inner folds for all tuning, early stopping, learned preprocessing, probability calibration and abstention threshold choice. Predictions from the outer folds are pooled **once per awakening**, with metrics and intervals clustered by participant. A single set of folds is shared across every comparator; save fold assignments and configuration. Do not describe this as an independent laboratory test.

**Conditional external evaluation:** After the internal protocol is frozen, train/tune/calibrate on Tononi participants only and evaluate once on all eligible LODE N2 E/NE cases; check identity overlap and the channel/label rules first. This assesses transfer from Tononi to LODE, not a symmetric leave-one-study-out experiment. Do not recalibrate on LODE labels. If harmonization fails, explicitly report external evaluation as unavailable and why.

| Comparator | Purpose | Implementation commitment |
|---|---|---|
| Training-fold class-prior / constant-probability baseline | Reveals what prevalence alone can achieve; N2 stage is constant in the primary study, so a stage-only classifier reduces to this. | Estimate E/NE prevalence on each outer training fold; never use test prevalence. |
| Simple non-EEG metadata sensitivity baseline | Check whether recording/protocol clues explain results. | Only fields available before the alarm and audited for meaning; no participant ID or report-derived fields. Omit if no defensible variable exists. |
| Band-power logistic regression | Interpretable conventional EEG reference. | Prespecified frequency bands per channel/region; training-fold standardization and class weighting/tuning only inside inner folds. |
| Compact 1D CNN | Test whether raw time structure helps beyond band power. | Only if the audited channel layout, fixed window and sample count permit it; a small regularized model with an explicit parameter count, early stopping on grouped validation, no large transformer. |
| Calibrated/selective prediction | Test whether uncertain predictions can be flagged. | Calibrate on grouped training-side validation predictions; report reliability/Brier and risk versus retained coverage at thresholds chosen without outer-test labels. |

Fit the binary classifiers on E/NE. For exploratory three-class analysis, **train a new three-output model** on E/WR/NE using the same participant folds and safeguards. A binary classifier's assignments for WR are a descriptive probe, not evidence of three-class prediction. If WR counts per fold after exclusions are too sparse, report only descriptive WR scores and confusion, explicitly declining three-class inference.

## 5. Measures, graphs and uncertainty

The unit for every main result is the **awakening**; uncertainty is resampled by **participant**, carrying all of that person's awakenings together. Report the number of participants, awakenings, class counts and exclusions per split/study. Main binary measures: macro F1, balanced accuracy, E and NE precision/recall, AUROC, positive-class AUPRC with its prevalence, Brier score and calibration plot. For three classes: macro F1, balanced accuracy, per-class precision/recall, confusion matrix and one-versus-rest AUROC/AUPRC only when denominators are adequate. State how undefined metrics are handled. Never substitute plain accuracy for the above.

Use participant-cluster bootstrap intervals (proposed 2,000 replicates with a fixed seed) over out-of-fold awakening predictions; compare models with paired resampling of the same participants. If a resample lacks a required class, omit that replicate for that metric and report the number retained. Show per-participant scores or counts so a few prolific people do not silently dominate. External LODE intervals resample its participants separately. For within-study folds and confidence intervals, state that repeated folds share training data and that bootstrap uncertainty does not cure design bias.

Planned figures: (1) the eligibility/class-count funnel by study and participant; (2) fold and per-class counts; (3) held-out confusion matrix with raw counts and row-normalized rates; (4) precision-recall curve with class prevalence; (5) calibration/reliability and Brier score; (6) participant-level distribution and the conditional external result. Every graph will name its denominator and test population. A heatmap or channel attribution, if later made, is exploratory and cannot prove a brain mechanism.

Ablations, in priority order: band-power versus CNN; pre-alarm timing window (one prespecified earlier interval); small montage versus a second prespecified montage if physically compatible; artifact/EOG sensitivity; and training-site versus held-out-site performance if the gate passes. Avoid broad searches across channels, windows or architecture variants.

## 6. Confounds, failure criteria and interpretation

Sleep stage is held at N2. Remaining confounds include subject-specific EEG signatures, recording hardware/reference, site-specific label questions, awakening timing, arousal, muscle/eye activity, age and artifact rejection. WR can reflect memory failure or uncertainty; it must not be interpreted as an objective intermediate brain state. Use participant grouping, a site baseline where meaningful, exclusion accounting and sensitivity analyses to test these risks, but do not imply they are eliminated.

A result supports only the narrow claim supported by its test set: within-study held-out-participant prediction, and possibly one-way held-out-site transfer. If the EEG model ties the prior or conventional baseline, if WR recall is poor, or if intervals are wide, report that plainly. Calibration means whether predicted probabilities agree with observed report frequencies on held-out observations; abstention means declining a prediction when its trained confidence rule is not met. Neither turns the classifier into a medical device.

## 7. Feasibility, milestones and stop conditions

A Windows 11 laptop with Python, 16 GB RAM and roughly 25 GB of free working storage is a **planning estimate** for downloading the approximately 7.83 GB compressed Tononi package plus temporary extraction/features. Stream one subject archive at a time where possible; fit band-power baselines on CPU. A modest GPU is optional for the small CNN, with CPU training capped by a fixed experiment budget. Record actual time, peak storage and hardware rather than promising runtime.

| Week | Reviewable deliverable | Gate |
|---|---|---|
| 1 | Raw-data manifest, exact subject/file/label/stage joins, timing and montage check, exclusion counts; lock v3 and splits | If reliable pre-alarm alignment or participant IDs fail, halt supervised modeling. |
| 2 | Grouped-fold class-prior and band-power baselines, fold-specific logs and plots | If E/NE support collapses after cleaning, narrow to a descriptive feasibility study. |
| 3 | Compact CNN only if baseline and signal gate pass; calibration and participant-cluster intervals | If CNN overfits or adds no value, retain the baseline as the main finding. |
| 4 | Conditional LODE external check, WR exploratory analysis, limitations and result-based abstract | If site compatibility/WR support fails, do not claim cross-site or three-class success. |

Implementation in Phase 3 begins with the manifest and baseline, **not** a final architecture selected from the catalogue. Raw data stay out of git. The README will explain acquisition, package permissions, pinned versions, configurable paths and one-command reproductions from locally obtained data. Before any submission, refresh the literature search and assess whether the measured contribution is substantial enough for a paper. No publication tier is promised.

## Primary references

- DREAM database and initial analyses: https://doi.org/10.1038/s41467-025-61945-1
- DREAM versioned catalogue: https://doi.org/10.26180/22133105
- Tononi v3 package: https://doi.org/10.26180/23306054.v3
- LODE package: https://doi.org/10.6084/m9.figshare.22147085.v1
- Closest N2 machine-learning study: https://doi.org/10.1093/sleepadvances/zpaf066
- Public generalization manuscript/repository (review status unverified): https://github.com/asim-v/dream-decoder-generalization-paper
