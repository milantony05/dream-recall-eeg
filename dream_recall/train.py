"""Grouped, nested logistic EEG baseline and prior-only comparator."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import figures, participant_ci, scores, write_results
from .features import band_power

SEED = 42


def load(path: Path, three_class: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise ValueError("Missing preprocessing sidecar with timing review")
    timing = json.loads(sidecar.read_text(encoding="utf-8")).get("timing_evidence", {})
    if timing.get("alignment_verified") is not True:
        raise ValueError("Alarm alignment remains unverified; do not fit models on provisional windows")
    with np.load(path) as data:
        X, y, groups, cases = (data[k].copy() for k in ["X", "label", "subject_id", "case_id"])
    if X.ndim != 3 or X.shape[1:] != (16, 3840) or len(set(cases)) != len(cases):
        raise ValueError("Corrupted/duplicate processed observations")
    if not set(y) <= {"E", "WR", "NE"}:
        raise ValueError("Invalid report labels")
    keep = np.ones(len(y), bool) if three_class else y != "WR"
    return X[keep], y[keep], groups[keep], cases[keep]


def splits(y: np.ndarray, groups: np.ndarray, wanted: int) -> list[tuple[np.ndarray, np.ndarray]]:
    for folds in range(wanted, 2, -1):
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
        result = list(cv.split(np.zeros(len(y)), y, groups))
        if all(len(set(y[train])) == len(set(y)) and len(set(y[test])) == len(set(y))
               and not (set(groups[train]) & set(groups[test])) for train, test in result):
            return result
    raise ValueError("No 3+ participant-disjoint folds with every class on both sides")


def run(input_path: Path, output: Path, three_class: bool = False) -> None:
    X, y, groups, cases = load(input_path, three_class)
    labels = ["NE", "WR", "E"] if three_class else ["NE", "E"]
    if len(set(y)) != len(labels) or len(set(groups)) < 9:
        raise ValueError("Insufficient labeled participants/classes")
    features = band_power(X)
    outer = splits(y, groups, 5)
    prior = np.zeros((len(y), len(labels)))
    predicted = np.zeros_like(prior)
    calibrated = np.zeros_like(prior) if not three_class else None
    accepted = np.zeros(len(y), dtype=bool) if not three_class else None
    fold_info = []
    for fold, (train, test) in enumerate(outer):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Participant leakage")
        inner = splits(y[train], groups[train], 3)
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
        search = GridSearchCV(estimator, {"logisticregression__C": [0.01, 0.1, 1, 10]},
                              scoring="f1_macro", cv=inner, n_jobs=1, refit=True)
        search.fit(features[train], y[train])
        order = list(search.best_estimator_.classes_)
        predicted[test] = search.predict_proba(features[test])[:, [order.index(label) for label in labels]]
        prior[test] = [(y[train] == label).mean() for label in labels]
        if not three_class:
            # Cross-fitted inner predictions calibrate/choose coverage using outer TRAIN subjects only.
            base = clone(search.best_estimator_)
            oof = cross_val_predict(base, features[train], y[train], cv=inner,
                                    method="predict_proba", n_jobs=1)
            inner_order = list(np.unique(y[train]))
            raw_e = oof[:, inner_order.index("E")]
            eps = 1e-6
            logit = lambda a: np.log(np.clip(a, eps, 1-eps) / (1-np.clip(a, eps, 1-eps)))
            calibrator = LogisticRegression(random_state=SEED).fit(logit(raw_e).reshape(-1, 1), (y[train] == "E").astype(int))
            inner_cal = calibrator.predict_proba(logit(raw_e).reshape(-1, 1))[:, 1]
            test_e = calibrator.predict_proba(logit(predicted[test, labels.index("E")]).reshape(-1, 1))[:, 1]
            calibrated[test, labels.index("E")] = test_e
            calibrated[test, labels.index("NE")] = 1 - test_e
            # Abstain on 20% lowest training-side confidence; no test-label choice.
            threshold = float(np.quantile(np.max(np.stack([inner_cal, 1-inner_cal], axis=1), axis=1), 0.20))
            accepted[test] = np.maximum(test_e, 1-test_e) >= threshold
        fold_info.append({"fold": fold, "train_participants": sorted(set(groups[train])),
                          "test_participants": sorted(set(groups[test])),
                          "train_counts": {label: int((y[train] == label).sum()) for label in labels},
                          "test_counts": {label: int((y[test] == label).sum()) for label in labels},
                          "chosen_C": search.best_params_["logisticregression__C"],
                          "confidence_threshold": threshold if not three_class else None})
    output.mkdir(parents=True, exist_ok=True)
    saved = dict(case_id=cases, subject_id=groups, label=y, probability=predicted,
                 prior_probability=prior, class_order=np.array(labels))
    if not three_class:
        saved.update(calibrated_probability=calibrated, accepted=accepted)
    np.savez_compressed(output / "out_of_fold.npz", **saved)
    result = {"source": str(input_path), "model": "band-power logistic regression", "seed": SEED,
              "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "sklearn": sklearn.__version__}, "folds": fold_info,
              "eeg": scores(y, predicted, labels), "prior": scores(y, prior, labels),
              "participant_bootstrap": participant_ci(y, predicted, groups, labels, other=prior, seed=SEED)}
    if not three_class:
        result["calibrated"] = scores(y, calibrated, labels)
        result["selective"] = {"retained": int(accepted.sum()), "coverage": float(accepted.mean()),
                               "metrics_on_retained": scores(y[accepted], calibrated[accepted], labels)
                               if len(set(y[accepted])) == 2 else None}
    write_results(output / "metrics.json", result)
    figures(y, calibrated if not three_class else predicted, labels, output)
    print(json.dumps({"n": len(y), "participants": len(set(groups)), "eeg": result["eeg"],
                      "prior": result["prior"]}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/processed/tononi-v3.npz"))
    p.add_argument("--out", type=Path, default=Path("outputs/binary_baseline"))
    p.add_argument("--three-class", action="store_true", help="Separate exploratory E/WR/NE fit")
    args = p.parse_args()
    run(args.input, args.out, args.three_class)


if __name__ == "__main__":
    main()
