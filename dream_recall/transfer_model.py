"""One-way Tononi-to-LODE evaluation with a channel-pooled neural network."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import participant_ci, scores, write_results
from .train import splits

LABELS = ["NE", "E"]


def pooled(f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.stack([np.concatenate((np.median(f[i, mask[i]], axis=0),
                                     np.diff(np.percentile(f[i, mask[i]], [25, 75], axis=0), axis=0)[0]))
                     for i in range(len(f))])


def run(path: Path, out: Path) -> dict:
    import torch
    from torch import nn

    torch.set_num_threads(4)
    torch.manual_seed(42)
    with np.load(path) as data:
        sf, sm, sy, sg = (data[k].copy() for k in ("source_F", "source_mask", "source_label", "source_subject"))
        tf, tm, ty, tg = (data[k].copy() for k in ("target_F", "target_mask", "target_label", "target_subject"))
    if sf.shape[1:] != (16, 5) or tf.shape[1:] != (16, 5) or not sm.all() or not (2 <= tm.sum(axis=1)).all():
        raise ValueError("Channel features or masks invalid")
    if set(sg) & set(tg) or set(sy) != set(LABELS) or set(ty) != set(LABELS):
        raise ValueError("Participant overlap or invalid labels")

    class ChannelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
            self.head = nn.Linear(16, 2)

        def forward(self, x, mask):
            z = self.encoder(x)
            return self.head((z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))

    folds = splits(sy, sg, 5)
    source_oof = {name: np.zeros((len(sy), 2)) for name in ("prior", "logistic", "channel_net")}
    target_ensemble = {name: [] for name in source_oof}
    source_summary = pooled(sf, sm)
    target_summary = pooled(tf, tm)
    for fold, (train, test) in enumerate(folds):
        if set(sg[train]) & set(sg[test]):
            raise AssertionError("Participant leakage")
        prior = np.array([(sy[train] == label).mean() for label in LABELS])
        source_oof["prior"][test] = prior
        target_ensemble["prior"].append(np.tile(prior, (len(ty), 1)))
        logistic = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1, max_iter=1000))
        logistic.fit(source_summary[train], sy[train])
        order = list(logistic.classes_)
        source_oof["logistic"][test] = logistic.predict_proba(source_summary[test])[:, [order.index(c) for c in LABELS]]
        target_ensemble["logistic"].append(logistic.predict_proba(target_summary)[:, [order.index(c) for c in LABELS]])

        # The scaler sees source-training participants only; padded channels stay masked.
        mean = sf[train].mean(axis=(0, 1))
        scale = np.maximum(sf[train].std(axis=(0, 1)), 1e-3)
        xs = torch.tensor((sf - mean) / scale, dtype=torch.float32)
        xt = torch.tensor((tf - mean) / scale, dtype=torch.float32)
        torch.manual_seed(42 + fold)
        model = ChannelNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        weights = torch.tensor([len(train) / (2 * (sy[train] == label).sum()) for label in LABELS], dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        rng = np.random.default_rng(42 + fold)
        for _ in range(40):
            for batch in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train) / 16)))):
                selected = torch.zeros((len(batch), 16), dtype=torch.float32)
                for j in range(len(batch)):
                    k = int(rng.integers(2, 17))
                    selected[j, rng.choice(16, size=k, replace=False)] = 1
                truth = torch.tensor([LABELS.index(v) for v in sy[batch]])
                optimizer.zero_grad()
                loss = loss_fn(model(xs[batch], selected), truth)
                loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            source_oof["channel_net"][test] = torch.softmax(model(xs[test], torch.tensor(sm[test], dtype=torch.float32)), -1).numpy()
            target_ensemble["channel_net"].append(torch.softmax(model(xt, torch.tensor(tm, dtype=torch.float32)), -1).numpy())

    out.mkdir(parents=True, exist_ok=True)
    result = {"design": "fixed 5-fold source participant CV; average five source-trained models on untouched LODE",
              "source": {"n": len(sy), "participants": len(set(sg)),
                         "classes": {str(k): int(v) for k, v in zip(*np.unique(sy, return_counts=True))}},
              "target": {"n": len(ty), "participants": len(set(tg)),
                         "classes": {str(k): int(v) for k, v in zip(*np.unique(ty, return_counts=True))}},
              "models": {}, "torch": torch.__version__}
    for name in source_oof:
        p = source_oof[name]
        q = np.mean(target_ensemble[name], axis=0)
        result["models"][name] = {"source_oof": scores(sy, p, LABELS), "target_external": scores(ty, q, LABELS),
                                  "source_ci": participant_ci(sy, p, sg, LABELS, repeats=1000),
                                  "target_ci": participant_ci(ty, q, tg, LABELS, repeats=1000)}
    write_results(out / "metrics.json", result)
    print(json.dumps({name: {"source_auroc": v["source_oof"]["auroc"],
                              "target_auroc": v["target_external"]["auroc"]}
                      for name, v in result["models"].items()}, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/processed/transfer.npz"))
    p.add_argument("--out", type=Path, default=Path("outputs/transfer"))
    a = p.parse_args(); run(a.input, a.out)


if __name__ == "__main__":
    main()
