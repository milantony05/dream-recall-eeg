# Dream recall from pre-awakening EEG

**Research status: Phase 3 implementation, before model training.** This project studies whether EEG recorded before an N2 awakening predicts a person's subsequent report of **recalled experience (E)** versus **no reported experience (NE)**. A third category, **experience without recalled content (WR, sometimes called a white dream)**, is exploratory. The target is a report after awakening. The system does not identify dream content or establish whether someone was conscious.

This is a small, reproducible undergraduate research project. It emphasizes participant-held-out testing, class balance, calibration and failure analysis. See the [data/literature audit](docs/phase1_feasibility_audit.md) and [research specification](docs/phase2_research_specification.md) for the rationale and the scope of claims. Existing DREAM studies have already classified binary reports; this repository does not claim a first dream classifier.

## Data and current gate

The primary source is the open [DREAM catalogue v9](https://doi.org/10.26180/22133105) and [Tononi Serial Awakenings v3](https://doi.org/10.26180/23306054.v3), **set 13, amendment 2**. The catalogue and the package's own `Records.csv` each describe **287 N2 awakenings from 39 participants**: 124 E, 56 WR and 107 NE. These are pre-quality-check counts. The primary binary task has 231 E/NE candidates. No model performance is reported yet.

Only a six-record participant archive has been inspected at signal level so far. Its EDF files vary between 257 numbered signals and 257 plus an extra `Status` signal; channel 81 is flat in the inspected intervals. The catalogue's EEG count should not be interpreted as the total number of EDF signals. The precise alarm alignment is still unresolved because `Records.csv` gives 120 seconds while sampled EDF headers give 120–121 seconds without an alarm marker. **The preprocessing command intentionally blocks until the [timing and integrity gate](docs/timing_and_integrity_gate.md) is resolved.** Full-archive integrity and signal quality remain to be checked. This is a methodological stop condition, not a software installation problem.

LODE (set 3) is a possible later **external** check, but its seven-channel, 250 Hz EEG and reporting protocol must be independently audited before any cross-study evaluation. Its 11 N2 WR candidates cannot justify a strong cross-lab three-class claim.

## Setup and acquisition

Python 3.11+ is recommended. On Windows 11, open a VS Code terminal in the repository:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For the optional CNN, install a PyTorch wheel appropriate to your CPU or GPU environment. A GPU is optional. Use a machine with roughly 25 GB of free working disk space for the full Tononi download and intermediate files; this is an estimate, not a measured peak. The code accepts `--raw`, `--manifest`, `--input` and `--out` paths rather than assuming a specific user directory.

The downloader uses the versioned Figshare API and verifies each downloaded file's MD5 and size. It does not fetch request-only data. Downloading only the small metadata/control package plus one subject is enough to inspect structure, but **not** to train:

```powershell
python -m dream_recall.acquire --root data/raw --tononi-subject 1
python -m dream_recall.manifest --raw data/raw --out outputs/manifest.csv
```

To audit the full approximately 7.83 GB package:

```powershell
python -m dream_recall.acquire --root data/raw --all-tononi
python -m dream_recall.manifest --raw data/raw --out outputs/manifest.csv --complete
python -m dream_recall.qc --raw data/raw --manifest outputs/manifest.csv --out outputs/channel_qc.json
```

Read the [timing review](docs/timing_and_integrity_gate.md), verify the source of the alarm anchor, and save a local `data/timing-evidence.json` as specified there. Only then:

```powershell
python -m dream_recall.preprocess --raw data/raw --manifest outputs/manifest.csv --timing-evidence data/timing-evidence.json --out data/processed/tononi-v3.npz
python -m dream_recall.train --input data/processed/tononi-v3.npz --out outputs/binary_baseline
python -m dream_recall.train --input data/processed/tononi-v3.npz --out outputs/exploratory_three_class --three-class
python -m dream_recall.cnn --input data/processed/tononi-v3.npz --out outputs/binary_cnn
```

The CNN command needs optional PyTorch. The binary baseline runs first. Raw EEG, report spreadsheets, processed arrays and results stay in ignored `data/` and `outputs/`; they are never committed by these scripts. The source datasets have their own terms and attribution requirements. Verify the license of each package before reuse or redistribution; cite DREAM and Tononi in research outputs.

## What the pipeline does

1. **Manifest:** joins catalogue and Tononi package rows by filename/case ID, verifies subject, class and N2 stage, then checks each EDF header. It explicitly accepts a documented optional `Status` channel but never treats it as EEG. A missing or inconsistent file blocks the complete audit.
2. **Preprocessing:** after timing verification, selects 16 spatially spread numbered scalp channels from the published SFP coordinates without using labels, excluding the observed constant channel 81. Extracts one nominal 30-second window per awakening, applies a 0.5–35 Hz band-pass, resamples to 128 Hz and applies fixed gross-artifact checks. The recorded reference is not silently changed. Rejections and the channel list are logged.
3. **Baseline:** computes five relative log band-power bands per channel and fits scaled, class-weighted logistic regression. A training-fold class-prior prediction is the stage-only comparator (N2 is constant). All tuning is inside participant-grouped training folds. A binary sigmoid calibration and an abstention threshold are fitted using training-side cross-fitted predictions; neither uses the outer test labels.
4. **CNN:** a compact fixed 1D convolutional model uses the same participant-disjoint outer folds. Input scaling and early stopping use training/validation participants only. It is an optional comparator, not a claimed improvement.
5. **Evaluation:** saves awakening-level out-of-fold probabilities, fold memberships, configuration, macro F1, balanced accuracy, class-specific precision/recall, binary AUROC/AUPRC, Brier/calibration, confusion and precision–recall plots. Confidence intervals resample whole participants, with paired differences against the prior comparator. The exploratory three-class fit is separate from the binary task.

**Interpretation:** An attractive curve does not prove a brain mechanism. Precision–recall depends on E prevalence. A high pooled score with low within-person or external transfer can reflect study protocol or identity cues. A calibrated probability means its predicted frequency agrees with held-out reports; abstention only marks uncertainty. Failures and undefined metrics should be reported as such.

## Repository map

| Path | Purpose |
|---|---|
| `docs/phase1_feasibility_audit.md` | Primary data/literature audit and go/no-go decision |
| `docs/phase2_research_specification.md` | Questions, labels, splits, baselines and stop rules |
| `docs/timing_and_integrity_gate.md` | Unresolved alarm alignment and full-file review |
| `dream_recall/acquire.py` | Pinned public catalogue/package download and hashes |
| `dream_recall/manifest.py` | Catalogue/package/EDF join and completeness check |
| `dream_recall/edf.py`, `qc.py`, `preprocess.py` | EDF reading, label-blind channel checks and controlled signal processing |
| `dream_recall/features.py`, `train.py`, `cnn.py` | Conventional and optional deep baselines |
| `dream_recall/evaluation.py` | Metrics, participant bootstrap and figures |
| `tests/` | Participant split and signal feature contracts |

### Citation

Wong et al., “A dream EEG and mentation database,” *Nature Communications* 16, 7495 (2025), https://doi.org/10.1038/s41467-025-61945-1. Tononi Serial Awakenings v3: https://doi.org/10.26180/23306054.v3. The N2 classification precedent is https://doi.org/10.1093/sleepadvances/zpaf066.
