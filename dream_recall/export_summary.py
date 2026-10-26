"""Export deidentified Phase 4 metrics without raw EEG or participant IDs."""

import argparse
import json
from pathlib import Path

from .evaluation import write_results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("results/phase4_metrics.json"))
    args = p.parse_args()
    names = ("binary_baseline", "timing_m1", "timing_p1", "binary_cnn", "three_class", "three_class_cnn")
    summarized = {}
    for name in names:
        d = json.loads((args.results / name / "metrics.json").read_text(encoding="utf-8"))
        summarized[name] = {key: d[key] for key in ("model", "seed", "eeg", "prior", "participant_bootstrap")}
        summarized[name]["fold_test_counts"] = [row["test_counts"] for row in d["folds"]]
        for key in ("calibrated", "selective", "software", "torch", "device", "torch_threads"):
            if key in d:
                summarized[name][key] = d[key]
    comparisons = {}
    for name in ("timing_comparison", "binary_model_comparison", "three_class_model_comparison"):
        d = json.loads((args.results / f"{name}.json").read_text(encoding="utf-8"))
        comparisons[name] = {key: value for key, value in d.items() if
                             key in ("n", "participants", "first_minus_second_participant_bootstrap",
                                     "primary_minus_shift_participant_bootstrap")}
    result = {"dataset": "Tononi Serial Awakenings v3 / DREAM set 13 amendment 2, N2 only",
              "timing": "Methods-supported nominal anchor; exact EDF alarm sample unmarked; primary seconds 85–115, shifts ±1 s",
              "note": "Participant-held-out, within-study results; no external test, exact alarm verification or dream-content decoding",
              "models": summarized, "paired_comparisons": comparisons}
    write_results(args.out, result)
    print(f"Saved deidentified metrics to {args.out}")


if __name__ == "__main__":
    main()
