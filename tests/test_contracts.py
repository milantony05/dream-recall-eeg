"""Contracts that would catch common EEG experiment failures without raw data."""

import unittest

import numpy as np

from dream_recall.features import band_power
from dream_recall.train import splits


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


if __name__ == "__main__":
    unittest.main()
