"""Paired participant bootstrap for two models on identical held-out awakenings."""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import participant_ci, scores, write_results


def compare(first: Path, second: Path, out: Path) -> dict:
    bundles = []
    fold_membership = []
    for folder in (first, second):
        with np.load(folder / "out_of_fold.npz") as archive:
            bundles.append({key: archive[key].copy() for key in
                            ("case_id", "subject_id", "label", "class_order", "probability")})
        metadata = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        fold_membership.append([row.get("test_participants", row.get("test_subjects"))
                                for row in metadata["folds"]])
    left, right = bundles
    for key in ("case_id", "subject_id", "label", "class_order"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"Unpaired model comparison: {key}")
    if fold_membership[0] != fold_membership[1]:
        raise ValueError("Outer folds differ between models")
    labels = left["class_order"].tolist()
    result = {"first": str(first), "second": str(second), "n": len(left["label"]),
              "participants": len(np.unique(left["subject_id"])),
              "first_metrics": scores(left["label"], left["probability"], labels),
              "second_metrics": scores(right["label"], right["probability"], labels),
              "first_minus_second_participant_bootstrap": participant_ci(
                  left["label"], left["probability"], left["subject_id"], labels,
                  other=right["probability"], seed=42)["paired_difference_ci95"]}
    write_results(out, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("first", "second", "out"):
        p.add_argument("--" + option, type=Path, required=True)
    args = p.parse_args()
    result = compare(args.first, args.second, args.out)
    print(json.dumps({"n": result["n"], "participants": result["participants"],
                      "first_macro_f1": result["first_metrics"]["macro_f1"],
                      "second_macro_f1": result["second_metrics"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
