# Phase 1 — DREAM feasibility and literature audit

Audit date: 22 October 2026. **Decision: no-go for a three-class, cross-study primary claim at present.** A focused N2, participant-held-out binary comparison is the course-project fallback. The three-class question is exploratory until raw EEG integrity, channel harmonization, label provenance, and held-out-study support are checked. No model has been trained and no performance result is claimed here.

## What the terms mean

EEG is a recording of changing electrical voltage at scalp electrodes. A sleep stage is a category assigned to a period of sleep: REM is rapid-eye-movement sleep; NREM includes N1, N2 and N3. A "white dream" is a report of having experienced something without recalling its content. It is a report category, not an objective measurement of consciousness. One awakening and its immediately preceding EEG are one labeled observation; several observations can belong to one participant.

## Sources and version boundary

- DREAM catalogue, version 9 (posted 30 October 2025): https://doi.org/10.26180/22133105. Downloaded `Datasets.csv` and `Data records.csv` from the article's public file API (file IDs 59170373 and 59170370). The catalogue is metadata; EEG packages are separate.
- DREAM project page explains that package URLs are in `Datasets.Data URL` and that absent URLs may require contacting the contributor: https://bridges.monash.edu/projects/The_Dream_EEG_and_Mentation_DREAM_database/158987
- Wong et al., *Nature Communications* (2025): https://doi.org/10.1038/s41467-025-61945-1. Its **initial-release** 20 studies, 505 participants and 2,643 awakenings must not be confused with the current catalogue.
- Tononi Serial Awakenings, current amendment 2 / package version 3: https://doi.org/10.26180/23306054.v3. Open download is 7.83 GB in a reports/metadata ZIP plus subject EEG ZIPs. Its previous version 2 contained 261 N2 awakenings from 36 participants; version 3 has 287 from 39.
- Catalogue metadata is CC BY 4.0; the checked Figshare EEG package pages also display CC BY 4.0. Check the individual package and contributor terms before redistribution. Raw EEG should remain outside git.

The catalogue CSV has **3,191 rows** because superseded amendments of study 13 are present alongside its latest amendment. Filtering `Latest amendment=TRUE` yields **2,669 awakening records, 20 studies, and 508 summed study-level participant counts**. The last number is not proved to be 508 unique people across studies. One study (SCANDataset, set 8, 85 records) is marked **Private** with no open Data URL; its note describes application for access. Nineteen latest-amendment studies (2,584 records) are marked Open. "Open" means the catalogue lists an open package; it does not certify every EEG file's integrity or common processing.

## Latest-amendment inventory

Counts below are catalogue **records**, before EEG quality review. E = recalled experience, WR = experience without recalled content, NE = no reported experience, Other = ambiguous/unknown. Set 8 is request-only. Rate and channels are catalogue values; a range denotes varying EEG channel counts. One study can contribute many awakenings from the same person.

| Set | Study | Access | Records | Participants | E | WR | NE | Other | EEG Hz / channels |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Zhang & Wamsley 2019 | Open | 308 | 28 | 238 | 0 | 70 | 0 | 400 / 58 |
| 2 | Oudiette N1Data | Open | 246 | 63 | 214 | 14 | 18 | 0 | 256 / 3 |
| 3 | LODE | Open | 190 | 28 | 118 | 14 | 58 | 0 | 250 / 7 |
| 4 | TWC USA | Open | 33 | 19 | 27 | 1 | 5 | 0 | 1000 / 6–21 |
| 5 | Donders | Open | 7 | 5 | 7 | 0 | 0 | 0 | 500 / 57–116 |
| 6 | REM Turku | Open | 134 | 18 | 123 | 8 | 3 | 0 | 500 / 24 |
| 7 | ChildrenDreaming | Open | 30 | 30 | 13 | 0 | 17 | 0 | 250 / 32 |
| 8 | SCANDataset | **Request** | 85 | 18 | 63 | 7 | 15 | 0 | 200 / 2–4 |
| 9 | Multiple awakenings | Open | 456 | 19 | 314 | 0 | 121 | 21 | 250 / 32 |
| 10 | Sleep Talking | Open | 22 | 12 | 11 | 1 | 10 | 0 | 250 / 32 |
| 11 | YoungAdults | Open | 65 | 65 | 40 | 0 | 25 | 0 | 128 / 21 |
| 12 | Older adults | Open | 40 | 40 | 18 | 0 | 22 | 0 | 256 / 19 |
| 13 | Tononi Serial Awakenings v3 | Open | 287 | 39 | 124 | 56 | 107 | 0 | 500 / 257 |
| 14 | Federal University of Rio Grande do Norte | Open | 41 | 41 | 25 | 0 | 16 | 0 | 1000 / 32 |
| 15 | Noreika Motor tDCS | Open | 49 | 10 | 49 | 0 | 0 | 0 | 500 / 16 |
| 16 | DATA1 | Open | 324 | 10 | 107 | 63 | 118 | 36 | 2000 / 25 |
| 17 | Aamodt evening | Open | 158 | 27 | 116 | 27 | 15 | 0 | 1000 / 62 |
| 18 | Aamodt morning | Open | 97 | 16 | 60 | 21 | 14 | 2 | 1000 / 62 |
| 19 | Kumral et al. | Open | 66 | 19 | 42 | 0 | 24 | 0 | 1000 / 128 |
| 20 | MEG Kyushu | Open | 31 | 1 | 24 | 4 | 0 | 3 | 1000 / 3 EEG (+ MEG) |

Set 20 is an MEG-led study with only one participant, so it is excluded from the proposed multi-participant EEG benchmark. The accessibility, rates and channel numbers are catalogue entries, not a claim that all channels have comparable names, references or usable quality.

## Count funnel and stage risk

| Catalogue filter | E | WR | NE | Ambiguous/unknown | Total |
|---|---:|---:|---:|---:|---:|
| Latest amendment, all access | 1,733 | 216 | 658 | 62 | 2,669 |
| Open packages only | 1,670 | 209 | 643 | 62 | 2,584 |
| Open, last stage N1/N2/N3/REM | 1,381 | 200 | 614 | 61 | 2,256 |
| Open, sleep stage, exact three labels | 1,381 | 200 | 614 | 0 | **2,195** |
| N2 only, open, exact three labels, excluding MEG Kyushu | **672** | **124** | **393** | 0 | **1,189** |

These are **metadata candidates, not verified model-ready EEG examples**. The open catalogue has 327 wake-scored records and one missing-stage record; 55 "No experience or without recall" and seven "Unknown" labels must not be forced into a class. For N2, 31 combined labels and one unknown are omitted. The 124 N2 WR observations come from only six open studies: Tononi 56, DATA1 26, Aamodt evening 15, LODE 11, Aamodt morning 9, Oudiette 7. Tononi alone supplies 45% of N2 WR. The Aamodt studies are from the same research group and must not automatically be counted as independent laboratories or participants.

Some apparent study names mislead: Oudiette N1Data has 209/246 records marked W, and only 14 N1 and 23 N2. Zhang & Wamsley has no WR class because participants were not explicitly asked to distinguish forgotten experience from no experience. REM Turku has 123 E but only three NE. A stage-only rule or a laboratory/protocol fingerprint could look deceptively predictive when stages and studies are pooled.

For a practical N2 core, Tononi supplies 124 E / 56 WR / 107 NE across 39 participants; LODE supplies 66 / 11 / 41 across 27 N2 participants. These are promising **audit targets**, not a combined training set yet. Tononi's 500 Hz, high-density montage and LODE's 250 Hz, seven-channel setup require a defensible shared-channel or feature strategy.

## Direct access checks performed

I fetched the public Figshare article metadata for 16 catalogue package URLs. The remaining open links include Tononi on Monash, the Rio Grande do Norte Monash DOI, and Kumral's separate DOI; the catalogue lists them as Open, but I have not exhaustively fetched all of those archives. Four small/reasonable public packages were downloaded into temporary local audit storage and inspected:

- LODE: a 133.6 MB Data.zip with 190 EDF files and participant-directory structure; an EDF header identifies a 140-second recording with 12 total signals (the catalogue counts seven EEG channels).
- Oudiette: 43.3 MB ZIP with 246 EDF files; sample EDF has 60 seconds, six total signals, three catalogue EEG channels.
- YoungAdults: 36.8 MB ZIP with 65 EDF files; sample EDF has 300 seconds and 21 signals.
- Tononi v3: its 384 KB reports/metadata ZIP contains 39 report spreadsheets; the first 174.8 MB subject ZIP contains six EDF epochs; a sample EDF header indicates 121 seconds and 257 signals.

This verifies downloadable signal files in these examples, **not** every awakening, aligned label, artifact status or EEG header across all studies. The Tononi v3 catalogue and N2 paper disagree in size because the paper used the earlier v2 release. Do not silently switch versions. Raw files and report spreadsheets inspected for access remain outside the project repository.

## Closest prior work and novelty

1. Wong et al. (2025) introduced DREAM and already classified E versus NE within NREM and REM using EEG features, including spectral power and catch22; six datasets were selected for their EEG analyses. https://doi.org/10.1038/s41467-025-61945-1
2. The *SLEEP Advances* N2 study used Tononi v2: 261 reports, 110 E / 50 WR / 101 NE, 36 participants, 256-channel 500 Hz EEG. It trained a segmented EEG/CSP/KNN **binary E-versus-NE** classifier and then assigned WR segments to either binary class; this is not a direct validated three-class classifier. Its reported high cross-validation performance must not be equated with held-out-participant or held-out-lab performance without checking grouping. https://doi.org/10.1093/sleepadvances/zpaf066
3. The blinded Dream Catcher experiment on 54 NREM reports found that spectral-power clustering grouped participant identity more readily than dreamfulness. https://doi.org/10.1093/nc/niaa006
4. A **public 2026 manuscript/reproduction package, with peer-review status not established in this audit**, directly examines participant-held-out and laboratory-held-out DREAM E-versus-NE classification. Its repository reports weak within-lab and transfer AUROCs for its selected feature families; these are that manuscript's findings, not ours. https://github.com/asim-v/dream-decoder-generalization-paper
5. A 2026 arXiv proposal about topological DREAM features states *target* AUCs, not demonstrated results; it does not establish an achieved benchmark. https://arxiv.org/abs/2607.09662

Therefore neither "first dream classifier" nor "first cross-lab DREAM test" is supportable. A potential narrow contribution would be a **carefully audited direct three-report-category assessment, especially E versus WR within N2**, with participant grouping, site controls, calibrated uncertainty and honest failure cases. It is a hypothesis, not established novelty or performance.

## Go/no-go gate before Phase 2

**Current verdict: no-go for a primary claim that a three-class EEG model generalizes across studies.** There are 124 candidate N2 WR reports, concentrated by site, and the raw cross-study channel/label compatibility has not been established. It remains a valid exploratory question. Begin with a narrow, stage-matched E-versus-NE research question using checked open datasets; report WR separately without re-labeling it as NE. A single-study Tononi v3 direct three-class result may be a course-project analysis, but cannot establish cross-laboratory generalization.

Before specifying architecture, verify every chosen file against catalogue filename, subject, class, last stage, duration, EEG channel names/reference and usable pre-awakening window. Inspect study-specific report questions and preprocessing. Freeze exclusions. Check participant identity across related studies. Establish that train/validation/test subjects are disjoint, and that a held-out-study test contains enough members of **each** target class. Keep any segmentation entirely within the awakening's participant split.

The evidence that could justify a stronger claim is a prespecified, participant-held-out N2 result with meaningful WR precision/recall and participant-cluster uncertainty, improvement over a study/stage metadata baseline and band-power model, credible calibration, and replicated results on at least one genuinely independent held-out study with sufficient WR observations. If those conditions fail, a rigorous negative or binary result is the honest output. No journal tier can be inferred from this audit.
