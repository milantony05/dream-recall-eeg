"""Paired participant-level comparison of the locked one-second timing shifts."""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import participant_ci, scores, write_results


def compare(primary: Path, earlier: Path, later: Path, out: Path) -> dict:
    paths = {"primary": primary, "earlier_minus_1s": earlier, "later_plus_1s": later}
    bundles = {}
    folds = {}
    for name, folder in paths.items():
        with np.load(folder / "out_of_fold.npz") as data:
            bundles[name] = {key: data[key].copy() for key in ("case_id", "subject_id", "label", "class_order", "probability")}
        folds[name] = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["folds"]
    base = bundles["primary"]
    for name, candidate in bundles.items():
        for field in ("case_id", "subject_id", "label", "class_order"):
            if not np.array_equal(candidate[field], base[field]):
                raise ValueError(f"Unpaired {field} in {name}")
        if [f["test_participants"] for f in folds[name]] != [f["test_participants"] for f in folds["primary"]]:
            raise ValueError(f"Outer test participants changed in {name}")
    labels = base["class_order"].tolist()
    if labels != ["NE", "E"]:
        raise ValueError("Timing comparison is prespecified for binary NE/E")
    result = {"n": len(base["label"]), "participants": len(np.unique(base["subject_id"])),
              "population": "same E/NE awakenings and outer participant folds for each shift",
              "metrics": {name: scores(b["label"], b["probability"], labels) for name, b in bundles.items()},
              "primary_minus_shift_participant_bootstrap": {}}
    for name in ("earlier_minus_1s", "later_plus_1s"):
        result["primary_minus_shift_participant_bootstrap"][name] = participant_ci(
            base["label"], base["probability"], base["subject_id"], labels,
            other=bundles[name]["probability"], seed=42)["paired_difference_ci95"]
    write_results(out, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("primary", "earlier", "later", "out"):
        p.add_argument("--" + option, type=Path, required=True)
    args = p.parse_args()
    result = compare(args.primary, args.earlier, args.later, args.out)
    print(json.dumps({name: {metric: value for metric, value in info.items() if metric in
                              ("macro_f1", "balanced_accuracy", "auroc", "auprc", "brier")}
                      for name, info in result["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
