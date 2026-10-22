"""Conventional, prespecified relative log band-power features."""

import numpy as np
from scipy.signal import welch

BANDS = ((0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30))


def band_power(X: np.ndarray, rate: int = 128) -> np.ndarray:
    if X.ndim != 3 or X.shape[1:] != (16, 3840) or not np.isfinite(X).all():
        raise ValueError("Expected finite (awakening, 16 channels, 3840 samples)")
    f, psd = welch(X, fs=rate, nperseg=rate * 4, axis=-1)
    total = np.trapezoid(psd[..., (f >= 0.5) & (f < 30)], f[(f >= 0.5) & (f < 30)], axis=-1)
    values = []
    for lo, hi in BANDS:
        mask = (f >= lo) & (f < hi)
        power = np.trapezoid(psd[..., mask], f[mask], axis=-1)
        values.append(np.log((power + 1e-12) / (total + 1e-12)))
    out = np.stack(values, axis=-1).reshape(len(X), -1)
    if out.shape != (len(X), 80) or not np.isfinite(out).all():
        raise ValueError("Feature shape/value error")
    return out
