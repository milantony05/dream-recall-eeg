"""Contracts that would catch common EEG experiment failures without raw data."""

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from dream_recall.features import band_power
from dream_recall.sensitivity import lock
from dream_recall.train import load, splits


class ExperimentContracts(unittest.TestCase):
    def test_participants_never_cross_outer_or_inner_fold(self):
        groups = np.repeat(np.arange(39).astype(str), 6)
        y = np.tile(np.array(["NE", "E", "WR", "E", "NE", "E"]), 39)
        for train, test in splits(y, groups, 5):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            for inner_train, validation in splits(y[train], groups[train], 3):
                self.assertFalse(set(groups[train][inner_train]) & set(groups[train][validation]))

    def test_known_alpha_signal_has_larger_alpha_relative_power(self):
        t = np.arange(3840) / 128
        signal = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        X = np.broadcast_to(signal, (2, 16, 3840)).copy()
        features = band_power(X).reshape(2, 16, 5)
        self.assertEqual(features.shape, (2, 16, 5))
        self.assertTrue(np.all(features[:, :, 2] > features[:, :, 0]))

    def test_bad_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            band_power(np.zeros((4, 15, 3840)))

    def test_undocumented_alarm_timing_blocks_model_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, X=np.zeros((2, 16, 3840)), label=["E", "NE"],
                     subject_id=["1", "2"], case_id=["a", "b"])
            path.with_suffix(".json").write_text(json.dumps({"timing_evidence": {
                "anchor": "record_duration", "timing_status": "unverified"}}))
            with self.assertRaisesRegex(ValueError, "Document the methods-supported nominal timing assumption"):
                load(path, False)

    def test_timing_comparison_rejects_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for shift in (-1, 0, 1):
                path = Path(directory) / f"shift_{shift}.npz"
                np.savez(path, case_id=["case1"], subject_id=["other" if shift == 1 else "p1"], label=["E"])
                path.with_suffix(".json").write_text(json.dumps({"window_shift_seconds": shift}))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "subject/label disagreement"):
                lock(paths, Path(directory) / "common.json")


if __name__ == "__main__":
    unittest.main()
