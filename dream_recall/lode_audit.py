"""File-level LODE v1 audit, before cross-laboratory modeling."""

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from .edf import header

LABELS = {"0": "NE", "1": "WR", "2": "E"}
CHANNELS = ("EEG Fp1-O1", "EEG Fp1-O2", "EEG Fp1-F7", "EEG F8-F7",
            "EEG F7-O1", "EEG F8-O2", "EEG Fp1-F8")


def audit(raw: Path) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader((raw / "Records.csv").open(encoding="utf-8-sig")))
    if len(rows) != 190:
        raise ValueError("Expected pinned LODE v1 record count 190")
    result = []
    with zipfile.ZipFile(raw / "Data.zip") as archive:
        edfs = {n for n in archive.namelist() if n.lower().endswith(".edf")}
        expected = {"Data/PSG/" + row["Filename"] for row in rows}
        if edfs != expected:
            raise ValueError("EDF inventory differs from Records.csv")
        for row in rows:
            name = "Data/PSG/" + row["Filename"]
            if not re.fullmatch(r"Data/PSG/DL\d{3}/DL\d{3}_night\d{2}\.edf", name):
                raise ValueError(f"Unexpected path: {name}")
            with archive.open(name) as stream:
                h = header(stream)
            if (h.duration != 140 or h.labels[:7] != CHANNELS or
                    any(unit != "uV" or samples != 250 for unit, samples in
                        zip(h.units[:7], h.samples_per_record[:7])) or
                    not h.patient.startswith(row["Subject ID"] + " ") or
                    row["Experience"] not in LABELS):
                raise ValueError(f"LODE header/record mismatch: {name}")
            status = "header_ok" if row["Case ID"] in h.patient else "case_id_mismatch"
            bad = {int(x) for x in re.findall(r"\d+", row["Remarks"] or "")}
            if not bad <= set(range(1, 8)):
                raise ValueError(f"Unexpected bad-channel remark: {name}")
            result.append({"filename": row["Filename"], "case_id": row["Case ID"],
                           "subject_id": row["Subject ID"], "label": LABELS[row["Experience"]],
                           "stage": row["Last sleep stage"], "status": status,
                           "good_channels": 7 - len(bad),
                           "bad_channels": ",".join(map(str, sorted(bad)))})
    if len({r["case_id"] for r in result}) != 190:
        raise ValueError("Duplicate LODE case ID")
    n2 = [r for r in result if r["stage"] == "2"]
    binary = [r for r in n2 if r["label"] in ("E", "NE") and r["status"] == "header_ok"]
    summary = {"source": "LODE v1 / DREAM set 3 amendment 0", "all_files": 190,
               "n2": len(n2), "n2_participants": len({r["subject_id"] for r in n2}),
               "n2_by_class": dict(Counter(r["label"] for r in n2)),
               "binary_n2": len(binary), "binary_participants": len({r["subject_id"] for r in binary}),
               "binary_with_fewer_than_two_good_channels": sum(r["good_channels"] < 2 for r in binary),
               "header_status": dict(Counter(r["status"] for r in result)),
               "bad_channel_count_binary": dict(Counter(r["good_channels"] for r in binary)),
               "channel_labels": CHANNELS, "rate_hz": 250, "edf_seconds": 140,
               "stage_note": "Only the last stage is provided; verify stage continuity for the chosen window.",
               "protocol_note": "Home dry-electrode headband; spontaneous morning awakening; report often recorded minutes later."}
    return result, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw/lode-v1"))
    p.add_argument("--out", type=Path, default=Path("outputs/lode_audit.json"))
    args = p.parse_args()
    rows, summary = audit(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = args.out.with_name("lode_manifest.csv")
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, rows[0]); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
