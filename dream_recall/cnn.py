"""Small 1D CNN, evaluated only on participant-held-out awakenings.

Requires optional PyTorch. The fixed model is deliberately modest (~tens of
thousands of parameters); it is not an architecture search.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from .evaluation import figures, participant_ci, scores, write_results
from .train import SEED, load, splits


def run(input_path: Path, output: Path, three_class: bool = False) -> None:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("Install optional PyTorch to run the compact CNN") from exc

    class CompactCNN(nn.Module):
        def __init__(self, classes: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(16, 24, 15, stride=4, padding=7), nn.GroupNorm(4, 24), nn.GELU(),
                nn.MaxPool1d(4),
                nn.Conv1d(24, 32, 11, stride=2, padding=5), nn.GroupNorm(4, 32), nn.GELU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.GroupNorm(4, 32), nn.GELU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, classes))

        def forward(self, batch):
            return self.net(batch)

    X, y, groups, cases = load(input_path, three_class)
    labels = ["NE", "WR", "E"] if three_class else ["NE", "E"]
    if len(set(y)) != len(labels):
        raise ValueError("Missing target class")
    outer = splits(y, groups, 5)
    probabilities = np.zeros((len(y), len(labels)))
    prior = np.zeros_like(probabilities)
    fold_log = []
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold, (train, test) in enumerate(outer):
        inner_train, val = splits(y[train], groups[train], 3)[0]
        fit = train[inner_train]
        valid = train[val]
        if (set(groups[fit]) & set(groups[valid])) or (set(groups[train]) & set(groups[test])):
            raise AssertionError("Participant leakage")
        # Only outer-training/inner-fit EEG determines input scaling.
        scale = np.std(X[fit], axis=(0, 2), keepdims=True).astype(np.float32)
        scale = np.maximum(scale, 1.0)
        def tensor(indices):
            return torch.from_numpy(np.clip(X[indices] / scale, -8, 8).astype(np.float32)).to(device)
        fit_x, val_x, test_x = tensor(fit), tensor(valid), tensor(test)
        fit_y = torch.tensor([labels.index(t) for t in y[fit]], device=device)
        val_y = torch.tensor([labels.index(t) for t in y[valid]], device=device)
        counts = np.bincount(fit_y.cpu().numpy(), minlength=len(labels))
        weights = torch.tensor(len(fit) / (len(labels) * counts), dtype=torch.float32, device=device)
        torch.manual_seed(SEED + fold)
        model = CompactCNN(len(labels)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_function = nn.CrossEntropyLoss(weight=weights)
        best, best_epoch, best_state, stale = float("inf"), 0, None, 0
        rng = np.random.default_rng(SEED + fold)
        for epoch in range(40):
            model.train()
            for batch in np.array_split(rng.permutation(len(fit)), max(1, int(np.ceil(len(fit) / 16)))):
                optimizer.zero_grad()
                idx = torch.as_tensor(batch, dtype=torch.long, device=device)
                loss = loss_function(model(fit_x[idx]), fit_y[idx])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(model(val_x), val_y))
            if validation_loss < best - 1e-4:
                best, best_epoch, stale = validation_loss, epoch + 1, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 7:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probabilities[test] = torch.softmax(model(test_x), dim=1).cpu().numpy()
        prior[test] = [(y[train] == label).mean() for label in labels]
        fold_log.append({"fold": fold, "fit_subjects": sorted(set(groups[fit])),
                         "validation_subjects": sorted(set(groups[valid])),
                         "test_subjects": sorted(set(groups[test])), "best_epoch": best_epoch,
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "test_counts": {label: int((y[test] == label).sum()) for label in labels}})
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "out_of_fold.npz", case_id=cases, subject_id=groups, label=y,
                        probability=probabilities, prior_probability=prior, class_order=np.array(labels))
    result = {"model": "compact 1D CNN", "source": str(input_path), "seed": SEED,
              "torch": torch.__version__, "device": str(device), "torch_threads": torch.get_num_threads(),
              "python": platform.python_version(),
              "folds": fold_log, "eeg": scores(y, probabilities, labels), "prior": scores(y, prior, labels),
              "participant_bootstrap": participant_ci(y, probabilities, groups, labels, other=prior, seed=SEED)}
    write_results(output / "metrics.json", result)
    figures(y, probabilities, labels, output)
    print(json.dumps({"n": len(y), "participants": len(set(groups)), "eeg": result["eeg"]}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/processed/tononi-v3.npz"))
    p.add_argument("--out", type=Path, default=Path("outputs/binary_cnn"))
    p.add_argument("--three-class", action="store_true", help="Separate exploratory E/WR/NE fit")
    args = p.parse_args()
    run(args.input, args.out, args.three_class)


if __name__ == "__main__":
    main()
