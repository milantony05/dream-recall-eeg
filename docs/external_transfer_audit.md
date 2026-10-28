# Independent LODE audit and one-way neural-network stress test

**28 October 2026. Decision: no-go for a journal claim of transferable dream-report prediction on these two studies.** This is an exploratory extension of the frozen Tononi Phase 4 result, not a replacement for it. No target labels were used to tune a model, channel choice, window, or preprocessing threshold after the external scores were seen.

## Source, permissions and report meaning

[LODE v1](https://doi.org/10.6084/m9.figshare.22147085.v1) is openly downloadable under CC BY 4.0. Its `ExperimentalDescription.txt` says volunteers used a DREEM dry-electrode headband at home for at least 14 nights and made a verbal report after spontaneous morning awakening. It explicitly distinguishes recalled experience (E), impression of experience without content recall (WR), and no experience (NE). The released EDF contains 140 seconds up to the estimated awakening. Some reports were recorded minutes later; the protocol allowed up to an hour after awakening for E/WR, with a different rule for NE. These differences from Tononi's laboratory prompted-awakening protocol affect interpretation, even though the labels have a similar three-way meaning.

All 190 LODE EDF files were present in the v1 archive and inspected. They carry seven 250 Hz, µV, **bipolar** EEG derivations (Fp1-O1, Fp1-O2, Fp1-F7, F8-F7, F7-O1, F8-O2, Fp1-F8) and five other signals. One REM file has a header case ID different from `Records.csv`; it is excluded from any model use. This audit found 118 N2 reports from 27 participants (66 E, 11 WR, 41 NE). The binary E/NE candidates are 107 from 26 participants, before our fixed signal check. LODE does not provide a full-window manually scored N2 hypnogram for our chosen interval; the available stage is the **last** stage.

The package's bad-channel remarks leave 2–7 usable EEG derivations per binary candidate. We applied the Tononi gross-signal rule without consulting LODE labels, dropped individually marked or corrupt channels, and required at least two. This excluded 24/107 candidate reports (15 E, 9 NE). **The external test is 83 reports from 24 participants: 51 E, 32 NE.** The high and unequal rejection rate is itself a limitation, not an accuracy improvement. Raw EDFs and case-level data remain outside Git.

## Frozen, limited cross-device experiment

A raw 16-channel Tononi CNN cannot directly ingest LODE's seven bipolar derivations. We therefore compared two deliberately small, channel-agnostic methods: (1) median and interquartile range of five relative log EEG frequency powers per good channel, followed by scaled logistic regression; (2) a channel-pooled neural network that applies the same small encoder to each channel's five powers, averages the unmasked embeddings, then predicts E/NE. The network uses label-blind random channel masking during Tononi training. Both share the same 0.5–35 Hz filtering, 128 Hz resampling and band definitions. Feature code agreed with the existing Tononi baseline to within 2.3e-7 on checked samples. These features avoid falsely treating LODE's bipolar channel 1 as equivalent to Tononi's numbered channel 1; they **do not eliminate** reference, montage, hardware or protocol differences.

Five participant-disjoint Tononi folds (194 E/NE reports, 35 participants) produced internal out-of-fold predictions. Each of the five resulting source-trained models predicted the completely unseen LODE cases, and their probabilities were averaged; no LODE label was used for fitting, calibration or model selection. Epochs from the same participant never crossed a source train/test fold. The target result has 95% intervals from resampling the 24 target participants, not individual EEG seconds. No LODE-tuned adaptation was attempted.

| Model | Tononi OOF AUROC | LODE AUROC [95% participant CI] | LODE balanced accuracy | LODE Brier ↓ |
|---|---:|---:|---:|---:|
| Source class prior | 0.485 | 0.500 [0.500, 0.500] | 0.500 | 0.242 |
| Pooled spectral logistic regression | 0.535 | 0.452 [0.296, 0.573] | 0.466 | 0.275 |
| Channel-masked pooled neural network | 0.584 | 0.447 [0.264, 0.592] | 0.535 | 0.250 |

The neural network has somewhat better *source* ranking than the pooled logistic model, but neither transfers to LODE; the external AUROC intervals include 0.5. The network's thresholded balanced accuracy of 0.535 does not contradict its AUROC of 0.447: a single threshold can look mildly favorable even when overall ranking is worse than chance, especially with 83 records. The prior-only Brier is at least as good as the neural alternatives. This result does not imply EEG carries no dream-report information; it says that this particular source-trained, channel-pooled representation did not demonstrate transport to this distinct home protocol.

## Publication decision

This single external failure is a useful **course-project negative result** and a useful engineering demonstration of rigorous data audit, masking, participant grouping and out-of-domain evaluation. It is not presently a convincing journal delta: recent [DREAM](https://doi.org/10.1038/s41467-025-61945-1) analyses and public generalization preprints already motivate these questions, while our target has only 83 usable reports, no exact common montage, differing awakening procedures and no direct validation of stage throughout the selected window. We cannot fix those problems by rerunning the same test, trying a bigger neural network, or picking the best window after viewing scores.

A defensible paper would require a **new prospective question and sufficient compatible data**—for example, multiple source laboratories with a held-out laboratory, fixed label and stage semantics, predeclared harmonization, and a neural method tested against simpler baselines without target-label tuning. That is a substantial new project with its own feasibility gate. If it cannot be assembled within the timeline, do not advertise this repository as a validated dream detector or a publication-ready paper.

The reproducible code is `dream_recall/lode_audit.py`, `transfer_data.py` and `transfer_model.py`; aggregate counts and exact metrics are in `results/lode_audit_summary.json` and `results/transfer_metrics.json`. External source: [LODE v1 dataset](https://doi.org/10.6084/m9.figshare.22147085.v1), and the archive's `ExperimentalDescription.txt` and `Records.csv`.
