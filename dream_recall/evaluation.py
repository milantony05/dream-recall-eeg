"""Awakening-level evaluation with participant-cluster uncertainty."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)


def scores(y: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict:
    if probabilities.shape != (len(y), len(labels)) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Probability shape or normalization error")
    prediction = np.array(labels)[np.argmax(probabilities, axis=1)]
    precision, recall, _, support = precision_recall_fscore_support(y, prediction, labels=labels, zero_division=0)
    result = {"n": len(y), "class_count": {name: int((y == name).sum()) for name in labels},
              "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
              "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
              "precision": dict(zip(labels, map(float, precision))),
              "recall": dict(zip(labels, map(float, recall))),
              "confusion": confusion_matrix(y, prediction, labels=labels).tolist()}
    if len(labels) == 2:
        truth = (y == "E").astype(int)
        if len(np.unique(truth)) < 2:
            raise ValueError("Both classes required for binary metrics")
        prob = probabilities[:, labels.index("E")]
        result.update(auroc=float(roc_auc_score(truth, prob)), auprc=float(average_precision_score(truth, prob)),
                      e_prevalence=float(truth.mean()), brier=float(brier_score_loss(truth, prob)))
    else:
        result["ovr"] = {}
        for i, label in enumerate(labels):
            truth = (y == label).astype(int)
            result["ovr"][label] = ({"auroc": float(roc_auc_score(truth, probabilities[:, i])),
                                       "auprc": float(average_precision_score(truth, probabilities[:, i])),
                                       "prevalence": float(truth.mean())} if len(np.unique(truth)) == 2 else None)
        result["multiclass_brier"] = float(np.mean(np.sum((probabilities - (y[:, None] == np.array(labels))) ** 2, axis=1)))
    return result


def participant_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, labels: list[str],
                   other: np.ndarray | None = None, repeats: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    targets = ["macro_f1", "balanced_accuracy"] + (["auroc", "auprc", "brier"] if len(labels) == 2 else [])
    targets += [f"recall_{name}" for name in labels] + [f"precision_{name}" for name in labels]
    distributions = {key: [] for key in targets}
    differences = {key: [] for key in targets} if other is not None else None
    for _ in range(repeats):
        draw = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([np.flatnonzero(groups == group) for group in draw])
        if len(np.unique(y[ix])) != len(labels):
            continue
        result = scores(y[ix], p[ix], labels)
        baseline = scores(y[ix], other[ix], labels) if other is not None else None
        for key in targets:
            value = (result["recall"][key[7:]] if key.startswith("recall_") else
                     result["precision"][key[10:]] if key.startswith("precision_") else result[key])
            distributions[key].append(value)
            if baseline is not None:
                other_value = (baseline["recall"][key[7:]] if key.startswith("recall_") else
                               baseline["precision"][key[10:]] if key.startswith("precision_") else baseline[key])
                differences[key].append(value - other_value)
    if any(len(v) < max(100, repeats // 2) for v in distributions.values()):
        raise ValueError("Too few class-complete participant bootstrap draws")
    out = {"requested_draws": repeats, "retained_draws": len(distributions[targets[0]]),
           "ci95": {key: list(map(float, np.quantile(value, [0.025, 0.975]))) for key, value in distributions.items()}}
    if differences is not None:
        out["paired_difference_ci95"] = {key: list(map(float, np.quantile(value, [0.025, 0.975]))) for key, value in differences.items()}
    return out


def figures(y: np.ndarray, p: np.ndarray, labels: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    result = scores(y, p, labels)
    cm = np.array(result["confusion"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel="Predicted report", ylabel="Actual report",
           title=f"Held-out awakenings (n={len(y)})")
    fig.tight_layout(); fig.savefig(destination / "confusion.png", dpi=160); plt.close(fig)
    if len(labels) == 2:
        truth = (y == "E").astype(int)
        prob = p[:, labels.index("E")]
        precision, recall, _ = precision_recall_curve(truth, prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, label=f"EEG; AUPRC={result['auprc']:.3f}")
        ax.axhline(truth.mean(), color="gray", linestyle="--", label=f"E prevalence={truth.mean():.3f}")
        ax.set(xlabel="E recall", ylabel="E precision", title="Participant-held-out precision–recall", xlim=(0, 1), ylim=(0, 1))
        ax.legend(); fig.tight_layout(); fig.savefig(destination / "precision_recall.png", dpi=160); plt.close(fig)
        observed, predicted = calibration_curve(truth, prob, n_bins=5, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(predicted, observed, "o-", label=f"Brier={result['brier']:.3f}")
        ax.set(xlabel="Mean predicted E probability", ylabel="Observed E fraction",
               title=f"Held-out calibration (n={len(y)})", xlim=(0, 1), ylim=(0, 1))
        ax.legend(); fig.tight_layout(); fig.savefig(destination / "calibration.png", dpi=160); plt.close(fig)


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
