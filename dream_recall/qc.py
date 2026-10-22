"""Label-blind signal checks for the complete selected-channel Tononi inventory."""

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .edf import header, indices_for_numbers, interval
from .preprocess import channels_from_sfp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--manifest", type=Path, default=Path("outputs/manifest.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs/channel_qc.json"))
    args = p.parse_args()
    rows = pd.read_csv(args.manifest, dtype={"subject_id": str})
    if len(rows) != 287 or not rows.status.isin(["header_ok", "header_variant_ok", "excluded_header_mismatch"]).all():
        raise ValueError("Complete 287-row header audit with declared exclusions required before channel QC")
    rows = rows[rows.status.isin(["header_ok", "header_variant_ok"])]
    selected = channels_from_sfp(args.raw.resolve())
    flat, peak = Counter(), Counter()
    rejected = []
    extras = Counter()
    for row in rows.itertuples():
        with zipfile.ZipFile(row.archive) as archive, archive.open(row.member) as stream:
            h = header(stream)
            extras[("status_present" if h.labels[-1] == "Status" else
                    "edf_annotations" if h.labels[-1] == "EDF Annotations" else "no_extra_signal")] += 1
            # An early 30-second sample checks channel integrity without an alarm anchor.
            x = interval(stream, h, indices_for_numbers(h, [i + 1 for i in selected]), 0, 30)
        bad_flat = [selected[i] + 1 for i in np.flatnonzero(np.ptp(x, axis=1) < 0.25)]
        bad_peak = [selected[i] + 1 for i in np.flatnonzero(np.max(np.abs(x), axis=1) > 500)]
        flat.update(bad_flat); peak.update(bad_peak)
        if bad_flat or bad_peak:
            rejected.append({"case_id": row.case_id, "subject_id": row.subject_id,
                             "flat_channels": bad_flat, "high_peak_channels": bad_peak})
    result = {"records": len(rows), "channels": [i + 1 for i in selected], "extra_signals": dict(extras),
              "flat_early_window_by_channel": dict(flat), "high_peak_early_window_by_channel": dict(peak),
              "records_with_flag": len(rejected), "flagged": rejected,
              "note": "Early-window QC is label blind and does not prove pre-alarm timing or validity of later windows."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "flagged"}, indent=2))


if __name__ == "__main__":
    main()
