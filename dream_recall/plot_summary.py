"""Plot held-out binary discrimination with participant bootstrap intervals."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("docs/figures/phase4_binary_comparison.png"))
    args = p.parse_args()
    folders = ["timing_m1", "binary_baseline", "timing_p1", "binary_cnn"]
    names = ["EEG: 1 s earlier", "EEG: primary", "EEG: 1 s later", "Compact CNN: primary"]
    items = [json.loads((args.results / folder / "metrics.json").read_text()) for folder in folders]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), layout="constrained")
    for ax, key, baseline in zip(axes, ("balanced_accuracy", "auroc"), (0.5, 0.5)):
        values = [item["eeg"][key] for item in items]
        bounds = [item["participant_bootstrap"]["ci95"][key] for item in items]
        ys = np.arange(len(items))[::-1]
        ax.axvline(baseline, color="#7f8c8d", linestyle="--", lw=1, label="Chance reference")
        for i, (y, value, (lo, hi)) in enumerate(zip(ys, values, bounds)):
            ax.errorbar(value, y, xerr=np.array([[value-lo], [hi-value]]), fmt="o", capsize=4,
                        color="#1266a4" if i == 1 else "#555d67", markersize=6)
        ax.set(xlim=(0.4, 0.75), ylim=(-0.7, len(items)-0.3), yticks=ys,
               yticklabels=names, xlabel=key.replace("_", " ").upper() if key == "auroc" else "Balanced accuracy")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_title("Thresholded classification")
    axes[1].set_title("Ranking discrimination")
    fig.suptitle("N2 E vs NE • 194 awakenings, 35 held-out participants\n95% participant bootstrap intervals", fontsize=11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
