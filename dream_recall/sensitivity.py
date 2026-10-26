"""Lock the same awakenings across prespecified timing windows before fitting."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def lock(paths: list[Path], output: Path) -> dict:
    if len(paths) != 3:
        raise ValueError("Supply exactly the earlier, primary and later preprocessed windows")
    windows = []
    for path in paths:
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path) as data:
            cases = data["case_id"].tolist()
            subjects = data["subject_id"].tolist()
            labels = data["label"].tolist()
        if len(cases) != len(set(cases)):
            raise ValueError(f"Duplicate case in {path}")
        windows.append((info["window_shift_seconds"], dict(zip(cases, zip(subjects, labels)))))
    if sorted(shift for shift, _ in windows) != [-1, 0, 1]:
        raise ValueError("Expected prespecified -1, 0, +1-second windows")
    common = set.intersection(*(set(rows) for _, rows in windows))
    if not common:
        raise ValueError("No common awakenings")
    for case in common:
        if len({rows[case] for _, rows in windows}) != 1:
            raise ValueError(f"Cross-window subject/label disagreement: {case}")
    primary = dict(windows)[0]
    result = {"case_ids": sorted(common), "n": len(common),
              "participants": len({primary[case][0] for case in common}),
              "classes": dict(Counter(primary[case][1] for case in common)),
              "windows": {str(shift): {"path": str(paths[i]), "eligible": len(rows)}
                          for i, (shift, rows) in enumerate(windows)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs=3, required=True, metavar="NPZ")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = lock(args.inputs, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "case_ids"}, indent=2))


if __name__ == "__main__":
    main()
