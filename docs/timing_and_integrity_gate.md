# Phase 3 data integrity and timing gate

**Integrity audit complete; exact alarm alignment unresolved (22 October 2026). No real model has been fitted.** The complete Tononi v3 package was downloaded with the pinned Figshare file size and MD5 checks. Its 287 catalogue rows match `Records.csv` by file and case ID, participant, report category and N2 stage. The manifest accounts for all 287 EDFs: 256 classic numbered-channel headers, 26 EDF+ variants with numbered-channel subsets and an annotation channel, and five declared incompatible EDFs for participant 23. Those five last 300 seconds and have a different 256-channel layout against the package's 120-second/257-channel metadata, so the [exclusion list](../config/edf_exclusions.csv) removes them (three E, two NE).

The 26 EDF+ variants have 216, 237 or 226 total signals rather than the catalogue's 257 EEG channels. Their selected numbered channels have 500 Hz and µV units, and the reader maps the actual EDF label rather than assuming file position. Their annotation signal contains timekeeping entries near the end in inspected samples, not a clear alarm marker. The [frozen 16-channel montage](../config/channel_selection.json) is the intersection-based, label-blind selection of scalp electrodes available in all 282 compatible files; channel 81 was excluded because it was constant in sampled participant-1 windows. The first 30 seconds of all 282 compatible EDFs were checked for flat or high-peak selected channels. The same fixed rule applied to a nominal later window rejects 38 records. The resulting **provisional** eligibility set contains 244 awakenings across 35 participants: 106 E, 50 WR, 88 NE. Rejection totals are 18 E, 6 WR, 19 NE including the five header exclusions. These losses may be associated with the report and should be disclosed in any eventual results.

## What timing evidence does and does not establish

[Siclari et al. (2017), Methods](https://www.nature.com/articles/nn.4545.pdf) describes extracting two minutes preceding each awakening. The [N2 reanalysis](https://doi.org/10.1093/sleepadvances/zpaf066) also describes two-minute pre-awakening epochs. Tononi v3 `Records.csv` lists 120 seconds, while many classic EDFs span 121 seconds. The EDF header does not independently identify the alarm sample, and the additional second has not been explained. It is reasonable to inspect seconds 85–115 from the start as a **nominal** `[-35,-5)` window against the 120-second record, but the intended exact pre-alarm interval has not been verified for this release. The five-second margin does not prove the alignment.

This matters because an alarm or post-awakening signal would answer a different question. Before model fitting, check original acquisition documentation, events, supplementary methods, or obtain contributor clarification establishing the v3 record start/end relative to the alarm, and resolve any variable offset. If that changes the anchor, revise the window and protocol before scores are viewed. Preserve the timing uncertainty in any eventual paper.

The local, ignored `data/timing-evidence.json` can record the **provisional** inference for signal engineering:

```json
{
  "anchor": "record_duration",
  "alignment_verified": false,
  "evidence": "Siclari et al. 2017 Methods describes two-minute pre-awakening extraction; Tononi v3 Records.csv says 120 seconds. This supports a nominal window, but the v3 alarm sample and extra EDF second are unresolved."
}
```

`preprocess` embeds this review verbatim in its sidecar. `train` and `cnn` call the same loader and refuse to fit while `alignment_verified` is absent or false. Mark it true only when the additional independent evidence actually establishes the v3 timing anchor; include the source, section, date and estimated offset/error in `evidence`, then regenerate the processed dataset. The gate expresses a scientific dependency, not an installation issue.
