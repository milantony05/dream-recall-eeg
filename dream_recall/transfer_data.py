"""Frozen montage-agnostic spectral inputs for Tononi-to-LODE stress test."""

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import welch

from .edf import header, interval
from .features import BANDS, band_power
from .lode_audit import audit
from .preprocess import convert


def spectrum(x: np.ndarray) -> np.ndarray:
    """Five relative log powers per channel, identical to the source feature definition."""
    f, psd = welch(x, fs=128, nperseg=512, axis=-1)
    total_mask = (f >= .5) & (f < 30)
    total = np.trapezoid(psd[..., total_mask], f[total_mask], axis=-1)
    values = []
    for lo, hi in BANDS:
        mask = (f >= lo) & (f < hi)
        power = np.trapezoid(psd[..., mask], f[mask], axis=-1)
        values.append(np.log((power + 1e-12) / (total + 1e-12)))
    return np.stack(values, axis=-1).astype(np.float32)


def make(source: Path, lode: Path, out: Path) -> dict:
    rows, audit_summary = audit(lode)
    with np.load(source) as data:
        selection = np.isin(data["label"], ["E", "NE"])
        sf = band_power(data["X"][selection]).reshape(-1, 16, 5).astype(np.float32)
        sy, sg, sc = (data[key][selection].copy() for key in ("label", "subject_id", "case_id"))
    features, masks, labels, subjects, cases, exclusions = [], [], [], [], [], []
    with zipfile.ZipFile(lode / "Data.zip") as archive:
        for row in rows:
            if row["stage"] != "2" or row["label"] not in ("E", "NE") or row["status"] != "header_ok":
                continue
            bad = {int(k) for k in row["bad_channels"].split(",") if k}
            chosen = [k for k in range(7) if k + 1 not in bad]
            with archive.open("Data/PSG/" + row["filename"]) as stream:
                h = header(stream)
                uv = interval(stream, h, chosen, 105, 135)
            good = (np.ptp(uv, axis=1) >= .25) & (np.max(np.abs(uv), axis=1) <= 500)
            uv = uv[good]
            if len(uv) < 2:
                exclusions.append({"case_id": row["case_id"], "label": row["label"], "reason": "fewer_than_two_good_channels"})
                continue
            f = spectrum(convert(uv, 250))
            padded = np.zeros((16, 5), np.float32)
            padded[:len(f)] = f
            mask = np.zeros(16, bool)
            mask[:len(f)] = True
            features.append(padded); masks.append(mask)
            labels.append(row["label"]); subjects.append(row["subject_id"]); cases.append(row["case_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, source_F=sf, source_mask=np.ones((len(sf), 16), bool),
                        source_label=sy, source_subject=sg, source_case=sc,
                        target_F=np.stack(features), target_mask=np.stack(masks),
                        target_label=np.array(labels), target_subject=np.array(subjects), target_case=np.array(cases))
    result = {"source_binary": len(sf), "source_participants": len(set(sg)),
              "target_binary": len(labels), "target_participants": len(set(subjects)),
              "target_by_class": dict(Counter(labels)), "target_signal_exclusions": exclusions,
              "lode_header_status": audit_summary["header_status"],
              "input": "five relative log frequency powers per good channel; padded channels masked",
              "target_window": "seconds 105-135 of 140; last-stage N2, no full-window hypnogram",
              "source_window": "seconds 85-115 of nominal 120; exact alarm sample unmarked"}
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/processed/tononi-v3.npz"))
    p.add_argument("--lode", type=Path, default=Path("data/raw/lode-v1"))
    p.add_argument("--out", type=Path, default=Path("data/processed/transfer.npz"))
    a = p.parse_args()
    print(json.dumps(make(a.source, a.lode, a.out), indent=2))


if __name__ == "__main__":
    main()
