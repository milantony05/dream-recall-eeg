"""Join Tononi v3 catalogue/package records to EDF headers, without model fitting."""

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from .edf import header

LABELS = {"0": "NE", "1": "WR", "2": "E"}


def _csv_from_zip(z: zipfile.ZipFile, suffix: str) -> list[dict]:
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"Expected exactly one {suffix}, got {names}")
    return list(csv.DictReader(io.TextIOWrapper(z.open(names[0]), encoding="utf-8-sig")))


def build(raw: Path, *, complete: bool = False) -> pd.DataFrame:
    cat = raw / "catalogue-v9" / "Data records.csv"
    root = raw / "tononi-v3"
    control = root / "Tononi Serial Awakenings-Part1-No_PSGs.zip"
    if not cat.exists() or not control.exists():
        raise FileNotFoundError("Run python -m dream_recall.acquire first")
    with cat.open(encoding="utf-8-sig", newline="") as f:
        current = [r for r in csv.DictReader(f) if r["Set ID"] == "13" and r["Amendment"] == "2"]
    with zipfile.ZipFile(control) as z:
        package = _csv_from_zip(z, "Records.csv")
        loc = [n for n in z.namelist() if n.endswith("HydroCelGSN256v10.sfp")]
        if len(loc) != 1:
            raise ValueError("Missing or duplicate electrode map")
    if len(current) != 287 or len(package) != 287:
        raise ValueError("This pipeline is pinned to Tononi v3's 287 catalogue/package rows")
    by_key = {(r["Filename"], r["Case ID"]): r for r in current}
    if len(by_key) != 287:
        raise ValueError("Duplicate catalogue awakening key")
    files = {}
    for archive in sorted(root.glob("*PSGs.zip")):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if member.lower().endswith(".edf"):
                    filename = Path(member).name
                    if filename in files:
                        raise ValueError(f"Duplicate EDF filename: {filename}")
                    files[filename] = (archive, member)
    rows = []
    for r in package:
        key = (r["Filename"], r["Case ID"])
        c = by_key.get(key)
        if c is None:
            raise ValueError(f"Package awakening absent from current catalogue: {key}")
        if c["Subject ID"] != r["Subject ID"] or c["Experience"] != {"0": "No experience", "1": "Without recall", "2": "Experience"}.get(r["Experience"]):
            raise ValueError(f"Subject or label disagreement: {key}")
        if r["Experience"] not in LABELS or r["Last sleep stage"] != "2" or c["Last sleep stage"] != "N2":
            raise ValueError(f"Unexpected class/stage: {key}")
        name = r["Filename"]
        if not re.fullmatch(r"s\d{2}_ep\d{2}\.edf", name):
            raise ValueError(f"Unsafe/unexpected EDF filename: {name}")
        if int(name[1:3]) != int(r["Subject ID"]):
            raise ValueError(f"Filename/participant mismatch: {key}")
        row = dict(study="Tononi-v3", filename=name, case_id=r["Case ID"], subject_id=r["Subject ID"],
                   label=LABELS[r["Experience"]], stage="N2", nominal_seconds=float(r["Duration"]),
                   catalogue_rate=int(c["EEG sample rate"]), catalogue_channels=int(c["Number of EEG channels"]),
                   archive="", member="", edf_seconds=None, edf_rate=None, edf_channels=None, status="missing_edf")
        if name in files:
            archive, member = files[name]
            with zipfile.ZipFile(archive) as z, z.open(member) as stream:
                h = header(stream)
            rate = h.samples_per_record[0] / h.record_seconds
            row.update(archive=str(archive.resolve()), member=member, edf_seconds=h.duration,
                       edf_rate=rate, edf_channels=len(h.labels), status="header_ok")
            # Some EDFs append a non-EEG Status signal. The catalogue counts EEG, not total EDF signals.
            if (rate != 500 or rate != row["catalogue_rate"] or row["catalogue_channels"] != 257
                    or len(h.labels) not in (257, 258) or (len(h.labels) == 258 and h.labels[-1] != "Status")):
                row["status"] = "header_mismatch"
            elif not all(h.labels[i] == f"Chan {i + 1}" and h.units[i] == "uV" for i in range(257)):
                row["status"] = "montage_mismatch"
            elif h.duration < row["nominal_seconds"] or h.duration - row["nominal_seconds"] > 2:
                row["status"] = "duration_mismatch"
            elif not (h.patient.startswith(r["Subject ID"] + " ") and r["Case ID"] in h.patient):
                row["status"] = "patient_mismatch"
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["subject_id", "case_id"])
    if len(result) != 287 or result.duplicated(["study", "case_id"]).any():
        raise ValueError("Manifest key mismatch")
    if complete and set(files) != {r["Filename"] for r in package}:
        raise ValueError("EDF inventory differs from package Records.csv")
    if complete and (result["status"] != "header_ok").any():
        raise ValueError(f"Incomplete/invalid raw archive inventory: {Counter(result['status'])}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("outputs/manifest.csv"))
    p.add_argument("--complete", action="store_true", help="Fail unless all 287 EDF headers match")
    args = p.parse_args()
    result = build(args.raw.resolve(), complete=args.complete)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    summary = {"source": "Tononi amendment 2 / article version 3", "rows": len(result),
               "classes": dict(Counter(result["label"])), "participants": result.subject_id.nunique(),
               "status": dict(Counter(result["status"])), "alarm_alignment": "NOT VERIFIED BY EDF HEADER"}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
