# Phase 3 data integrity and timing gate

**Integrity audit complete; nominal timing assumption supported by published methods, exact EDF alarm sample unresolved (22 October 2026).** The complete Tononi v3 package was downloaded with the pinned Figshare file size and MD5 checks. Its 287 catalogue rows match `Records.csv` by file and case ID, participant, report category and N2 stage. The manifest accounts for all 287 EDFs: 256 classic numbered-channel headers, 26 EDF+ variants with numbered-channel subsets and an annotation channel, and five declared incompatible EDFs for participant 23. Those five last 300 seconds and have a different 256-channel layout against the package's 120-second/257-channel metadata, so the [exclusion list](../config/edf_exclusions.csv) removes them (three E, two NE).

The 26 EDF+ variants have 216, 237 or 226 total signals rather than the catalogue's 257 EEG channels. Their selected numbered channels have 500 Hz and µV units, and the reader maps the actual EDF label rather than assuming file position. Their annotation signal contains timekeeping entries near the end in inspected samples, not a clear alarm marker. The [frozen 16-channel montage](../config/channel_selection.json) is the intersection-based, label-blind selection of scalp electrodes available in all 282 compatible files; channel 81 was excluded because it was constant in sampled participant-1 windows. The first 30 seconds of all 282 compatible EDFs were checked for flat or high-peak selected channels. The same fixed rule applied to a nominal later window rejects 38 records. The resulting **provisional** eligibility set contains 244 awakenings across 35 participants: 106 E, 50 WR, 88 NE. Rejection totals are 18 E, 6 WR, 19 NE including the five header exclusions. These losses may be associated with the report and should be disclosed in any eventual results.

## Timing evidence, working assumption, and limitation

[Siclari et al. (2017), Methods](https://www.nature.com/articles/nn.4545.pdf) describes extracting two minutes preceding each awakening. The [N2 reanalysis](https://doi.org/10.1093/sleepadvances/zpaf066) also describes two-minute pre-awakening epochs. Tononi v3 `Records.csv` lists 120 seconds, while many classic EDFs span 121 seconds. The EDF header does not independently identify the alarm sample, and the additional second has not been explained. Together, these sources support treating each v3 EDF as the extracted two-minute pre-awakening segment. Our working window is seconds 85–115 from nominal recording start: `[-35,-5)` relative to the package's 120-second record duration. This is a defensible, published-method-supported nominal alignment for a course project, but the EDF itself does not expose the exact alarm sample. The 120/121-second header difference leaves a one-second ambiguity in the nominal endpoint; the selected window is five seconds before the 120-second endpoint and six seconds before a 121-second endpoint.

This matters because an alarm or post-awakening signal would answer a different question. We can proceed without contributor clarification using this prespecified working assumption. Keep the endpoint ambiguity visible and, if the results are worth pursuing, run a one-second earlier/later window sensitivity analysis without choosing the better result post hoc. If new documentation contradicts the assumption, revise the window and protocol before interpreting scores. State this timing limitation in any report or paper.

The local, ignored `data/timing-evidence.json` records the methods-supported nominal assumption for reproducibility:

```json
{
  "anchor": "record_duration",
  "timing_status": "methods_supported_nominal",
  "anchor_uncertainty": "The original paper specifies two minutes before awakening, and Tononi v3 records 120 seconds. The EDF headers do not mark the alarm sample; some files span 121 seconds. The chosen 85–115 s interval is 5 s before the nominal 120 s endpoint (6 s for a 121 s file)."
}
```

`preprocess` stores the review in its sidecar. `train` and `cnn` require this timing status and uncertainty text, while allowing the published-method-supported nominal analysis to proceed. This is a documented methodological assumption, not sample-exact alarm verification.
