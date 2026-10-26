"""Filtering and length handling adapted from preprocessing cells 24 and 26."""
import numpy as np
from scipy import signal as sp_signal


def pad_or_crop(signal, target_length=1000, mode="center", rng=None):
    if mode not in {"center", "random"} or target_length < 1:
        raise ValueError("Invalid crop mode or target length")
    rng = rng if rng is not None else np.random.default_rng()
    length = signal.shape[-1]
    if length < target_length:
        left = (target_length - length) // 2 if mode == "center" else int(rng.integers(target_length - length + 1))
        return np.pad(signal, ((0, 0), (left, target_length - length - left)))
    start = (length - target_length) // 2 if mode == "center" else int(rng.integers(length - target_length + 1))
    return signal[:, start:start + target_length]


def preprocess_signal(signal, sampling_rate=100, target_length=1000, apply_filter=True):
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 12 or x.shape[1] < 32:
        raise ValueError(f"Expected 12 leads and >=32 samples, got {x.shape}")
    if not np.isfinite(x).all() or (x.std(axis=1) < 1e-8).all():
        raise ValueError("ECG contains nonfinite values or all leads are flat")
    if apply_filter:
        # Same 4th-order 0.5-40 Hz Butterworth bandpass, stable SOS form.
        sos = sp_signal.butter(4, [0.5, 40.0], btype="bandpass", fs=sampling_rate, output="sos")
        x = sp_signal.sosfiltfilt(sos, x, axis=-1)
        if 50 < sampling_rate / 2:
            b, a = sp_signal.iirnotch(50, 30, fs=sampling_rate)
            x = sp_signal.filtfilt(b, a, x, axis=-1)
    return pad_or_crop(x, target_length).astype(np.float32)


class SignalNormalizer:
    """Streaming population mean/std, fitted on processed training waveforms only."""
    def fit(self, signals):
        count, mean, m2 = 0, np.zeros((12, 1)), np.zeros((12, 1))
        for x in signals:
            x = np.asarray(x, dtype=np.float64)
            n = x.shape[1]
            delta = x.mean(axis=1, keepdims=True) - mean
            m2 += ((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1, keepdims=True) + delta ** 2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        if not count:
            raise ValueError("Cannot fit normalization without training signals")
        self.mean, self.std = mean, np.maximum(np.sqrt(m2 / count), 1e-8)
        return self

    def transform(self, x):
        if not hasattr(self, "mean"):
            raise RuntimeError("Fit the normalizer on training data first")
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls()
        obj.mean, obj.std = np.asarray(state["mean"]), np.asarray(state["std"])
        if obj.mean.shape != (12, 1) or obj.std.shape != (12, 1) or not np.isfinite(obj.mean).all() or not np.isfinite(obj.std).all() or (obj.std <= 0).any():
            raise ValueError("Invalid normalization state")
        return obj
