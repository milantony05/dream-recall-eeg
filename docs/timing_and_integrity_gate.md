# Timing and integrity gate before any model run

**Status: open.** The Tononi v3 package's `Records.csv` states 120 seconds for the inspected awakenings. Among the six EDFs inspected for participant 1, five span 121 seconds and one spans 120 seconds. Their headers have no explicit alarm annotation. Five also include an extra `Status` signal in addition to the 257 numbered channels; the sampled Status signal was constant zero near the end. Numbered channel 81 was constant in the sampled windows and is excluded from the fixed 16-sensor montage. These observations were made without class-dependent channel selection.

Evidence for the *nominal* alignment exists in the original [Siclari et al. (2017) methods](https://www.nature.com/articles/nn.4545.pdf), which describe extracting two minutes preceding each awakening, and the [N2 reanalysis](https://doi.org/10.1093/sleepadvances/zpaf066), which describes each EEG epoch as two minutes before awakening. Together with the 120-second package record, these support interpreting the first 120 seconds as pre-awakening. They do **not** explain each additional EDF second or independently mark the alarm sample in v3; that distinction must remain in the method and limitations.

The primary proposed input is seconds 85–115 relative to the beginning of a 120-second nominal recording, corresponding to `[-35,-5)` **only if** the package's `Duration` refers to the interval immediately preceding the alarm. The five-second margin does not itself establish this fact. Do not create the processed dataset until the record end's relation to alarm onset has been checked from original acquisition documentation, events, supplementary methods, or contributor clarification. Note whether a fixed offset varies across recordings and what uncertainty remains.

When evidence is obtained, save a **local, untracked** JSON file such as `data/timing-evidence.json` with:

```json
{
  "anchor": "record_duration",
  "evidence": "Specific source, page/section or contributor response establishing the alarm alignment for this release; include date and estimated maximum timing error."
}
```

The preprocessing CLI refuses to run without this review record. Writing a statement without evidence is not a valid scientific check. If the source reveals a different anchor, change the window code and specification before any model fitting, and document the decision.

The full-header manifest must report **287 header_ok** rows. For every EDF, also examine channel 81's signal behavior, the fixed selected sensors, reference/units and visible artifacts without consulting labels. Revise the frozen channel set before modeling if a systematic issue is found. Record quality exclusions by class and participant. A few inspected files cannot certify the complete dataset.
