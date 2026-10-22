"""Convert audited EDF to one fixed pre-awakening sample per report."""

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from .edf import header, interval

RATE = 128
WINDOW = (-35, -5)


def channels_from_sfp(raw: Path, n: int = 16) -> list[int]:
    """Spread scalp sensors using geometry alone (z >= 5); zero-based EDF positions."""
    archive = raw / "tononi-v3" / "Tononi Serial Awakenings-Part1-No_PSGs.zip"
    with zipfile.ZipFile(archive) as z:
        name = next(x for x in z.namelist() if x.endswith("HydroCelGSN256v10.sfp"))
        lines = z.read(name).decode("utf-8").splitlines()
    positions = {}
    for line in lines:
        fields = line.split()
        if fields[0].isdigit() and 1 <= int(fields[0]) <= 256:
            positions[int(fields[0]) - 1] = np.array([float(x) for x in fields[1:4]])
    if len(positions) != 256:
        raise ValueError("Expected 256 numbered SFP positions")
    # Channel 81 is a constant placeholder in the sampled v3 EDFs (physical range -1..1).
    # Confirm this fixed exclusion across every downloaded EDF in the complete audit.
    eligible = sorted(k for k, xyz in positions.items() if xyz[2] >= 5.0 and k != 80)
    selected = [max(eligible, key=lambda k: (positions[k][2], -k))]
    while len(selected) < n:
        pick = max((k for k in eligible if k not in selected),
                   key=lambda k: (min(np.linalg.norm(positions[k] - positions[s]) for s in selected), -k))
        selected.append(pick)
    return sorted(selected)


def convert(uv: np.ndarray, original_rate: int) -> np.ndarray:
    if uv.ndim != 2 or uv.shape[1] != 30 * original_rate:
        raise ValueError("Wrong pre-alarm window shape")
    sos = butter(4, [0.5, 35], btype="bandpass", fs=original_rate, output="sos")
    filtered = sosfiltfilt(sos, uv, axis=1)
    out = resample_poly(filtered, RATE, original_rate, axis=1).astype(np.float32)
    if out.shape != (uv.shape[0], 30 * RATE) or not np.isfinite(out).all():
        raise ValueError("Filtering/resampling produced invalid array")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--manifest", type=Path, default=Path("outputs/manifest.csv"))
    p.add_argument("--out", type=Path, default=Path("data/processed/tononi-v3.npz"))
    p.add_argument("--timing-evidence", type=Path, required=True,
                   help="Human-reviewed JSON: anchor='record_duration', evidence=<nonempty explanation>; see README")
    args = p.parse_args()
    decision = json.loads(args.timing_evidence.read_text(encoding="utf-8"))
    if decision.get("anchor") != "record_duration" or len(decision.get("evidence", "").strip()) < 30:
        raise ValueError("Pre-alarm record-duration anchor needs documented evidence, not an empty confirmation")
    df = pd.read_csv(args.manifest, dtype={"subject_id": str, "case_id": str})
    if len(df) != 287 or df.status.ne("header_ok").any() or df.duplicated(["study", "case_id"]).any():
        raise ValueError("Complete, unique, header-verified 287-row manifest required")
    selected = channels_from_sfp(args.raw.resolve())
    signals, kept, rejected = [], [], []
    for row in df.itertuples():
        with zipfile.ZipFile(row.archive) as z, z.open(row.member) as stream:
            h = header(stream)
            anchor = float(row.nominal_seconds)
            if h.duration < anchor or anchor + WINDOW[0] < 0 or anchor + WINDOW[1] > h.duration:
                raise ValueError(f"Unsafe interval for {row.filename}")
            uv = interval(stream, h, selected, anchor + WINDOW[0], anchor + WINDOW[1])
        # Fixed, label-blind corruption rules. Report per-class rejection counts after processing.
        peak = np.max(np.abs(uv), axis=1)
        span = np.ptp(uv, axis=1)
        if np.any(peak > 500) or np.any(span < 0.25):
            rejected.append((row.case_id, row.label, "peak_or_flat"))
            continue
        signals.append(convert(uv, 500))
        kept.append((row.case_id, row.subject_id, row.label))
    if not kept:
        raise ValueError("All windows rejected")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".part")
    with temporary.open("wb") as f:
        np.savez_compressed(f, X=np.stack(signals), case_id=np.array([r[0] for r in kept]),
                            subject_id=np.array([r[1] for r in kept]), label=np.array([r[2] for r in kept]),
                            channel=np.array([i + 1 for i in selected]))
    with np.load(temporary) as check:
        if check["X"].shape != (len(kept), len(selected), 3840):
            raise ValueError("Saved dataset failed verification")
    temporary.replace(args.out)
    sidecar = {"source": "Tononi-v3", "window_relative_to_nominal_end_seconds": list(WINDOW),
               "timing_evidence": decision, "channel_numbers": [i + 1 for i in selected],
               "rate_hz": RATE, "accepted": len(kept), "accepted_by_class": dict(Counter(r[2] for r in kept)),
               "excluded": [{"case_id": a, "label": b, "reason": c} for a, b, c in rejected]}
    args.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in sidecar.items() if k != "excluded"}, indent=2))


if __name__ == "__main__":
    main()
