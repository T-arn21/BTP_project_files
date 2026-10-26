"""Extracted from preprocessing notebook cell 28; opt-in training transforms."""
import copy
from typing import Dict
import numpy as np
import scipy.signal as sp_signal
from .signal import pad_or_crop

DEFAULT_PARAMS = {'gaussian_noise': {'prob': 0.5, 'std': 0.05}, 'amplitude_scaling': {'prob': 0.5, 'scale_range': (0.8, 1.2)}, 'random_time_shift': {'prob': 0.3, 'max_shift_frac': 0.1}, 'random_masking': {'prob': 0.3, 'mask_frac': 0.1, 'n_masks': 2}, 'random_crop': {'prob': 0.3, 'crop_frac': 0.9}, 'lead_dropout': {'prob': 0.2, 'max_leads_dropped': 2}, 'time_stretch': {'prob': 0.2, 'stretch_range': (0.9, 1.1)}, 'random_jitter': {'prob': 0.3, 'sigma': 0.02}, 'random_scaling': {'prob': 0.3, 'scale_range': (0.9, 1.1)}}

class ECGAugmentation:
    '''Configurable, reusable stochastic augmentation pipeline for 12-lead ECG signals.

    Every augmentation is independently toggled by its own probability so this
    single class serves both light supervised fine-tuning augmentation and
    aggressive SSL pretext-task augmentation (see ``DEFAULT_PARAMS``).

    Parameters
    ----------
    params : dict, optional
        Per-augmentation ``{"prob": ..., <hyperparams>}`` dict. Defaults to
        ``DEFAULT_PARAMS``.
    seed : int, optional
        Seed for this augmenter's private RNG (keeps augmentation reproducible
        independent of any global RNG state).
    '''

    def __init__(self, params: Dict[str, dict] = None, seed: int = None):
        self.params = copy.deepcopy(DEFAULT_PARAMS if params is None else params)
        self.rng = np.random.default_rng(seed)

    def _roll(self, name: str) -> bool:
        return self.rng.random() < self.params[name]["prob"]

    # ---- individual augmentations -------------------------------------------
    def gaussian_noise(self, signal: np.ndarray) -> np.ndarray:
        '''Add per-sample Gaussian noise scaled to each lead's own std.'''
        p = self.params["gaussian_noise"]
        std = signal.std(axis=-1, keepdims=True) * p["std"]
        return signal + self.rng.normal(0, 1, size=signal.shape) * std

    def amplitude_scaling(self, signal: np.ndarray) -> np.ndarray:
        '''Scale the whole signal by one random factor (simulates electrode-contact variance).'''
        lo, hi = self.params["amplitude_scaling"]["scale_range"]
        return signal * self.rng.uniform(lo, hi)

    def random_time_shift(self, signal: np.ndarray) -> np.ndarray:
        '''Circularly shift the signal along the time axis by a random offset.'''
        max_frac = self.params["random_time_shift"]["max_shift_frac"]
        max_shift = int(signal.shape[-1] * max_frac)
        if max_shift == 0:
            return signal
        shift = self.rng.integers(-max_shift, max_shift + 1)
        return np.roll(signal, shift, axis=-1)

    def random_masking(self, signal: np.ndarray) -> np.ndarray:
        '''Zero out one or more short random time segments ("cutout" for time series).'''
        p = self.params["random_masking"]
        out = signal.copy()
        seg_len = max(1, int(signal.shape[-1] * p["mask_frac"]))
        for _ in range(p["n_masks"]):
            start = self.rng.integers(0, max(1, signal.shape[-1] - seg_len))
            out[:, start:start + seg_len] = 0.0
        return out

    def random_crop(self, signal: np.ndarray) -> np.ndarray:
        '''Crop a random contiguous sub-window then zero-pad back to the original length.'''
        frac = self.params["random_crop"]["crop_frac"]
        crop_len = int(signal.shape[-1] * frac)
        cropped = pad_or_crop(signal, target_length=crop_len, mode="random", rng=self.rng)
        return pad_or_crop(cropped, target_length=signal.shape[-1], mode="center")

    def lead_dropout(self, signal: np.ndarray) -> np.ndarray:
        '''Zero out 1-N randomly chosen leads, exploiting inter-lead physiological redundancy.'''
        p = self.params["lead_dropout"]
        n_drop = self.rng.integers(1, p["max_leads_dropped"] + 1)
        out = signal.copy()
        drop_idx = self.rng.choice(signal.shape[0], size=min(n_drop, signal.shape[0]), replace=False)
        out[drop_idx, :] = 0.0
        return out

    def time_stretch(self, signal: np.ndarray) -> np.ndarray:
        '''Resample the signal by a random stretch factor, then pad/crop back to original length.'''
        lo, hi = self.params["time_stretch"]["stretch_range"]
        factor = self.rng.uniform(lo, hi)
        new_len = max(2, int(signal.shape[-1] * factor))
        stretched = sp_signal.resample(signal, new_len, axis=-1)
        return pad_or_crop(stretched.astype(np.float32), target_length=signal.shape[-1], mode="center")

    def random_jitter(self, signal: np.ndarray) -> np.ndarray:
        '''Add small i.i.d. Gaussian jitter, independent of each lead's amplitude scale.'''
        sigma = self.params["random_jitter"]["sigma"]
        return signal + self.rng.normal(0, sigma, size=signal.shape)

    def random_scaling(self, signal: np.ndarray) -> np.ndarray:
        '''Apply an independent random scale factor per lead (unlike global amplitude_scaling).'''
        lo, hi = self.params["random_scaling"]["scale_range"]
        factors = self.rng.uniform(lo, hi, size=(signal.shape[0], 1))
        return signal * factors

    # ---- pipeline -------------------------------------------------------------
    _OP_ORDER = ["random_crop", "time_stretch", "random_time_shift", "lead_dropout",
                 "random_masking", "amplitude_scaling", "random_scaling",
                 "gaussian_noise", "random_jitter"]

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        '''Apply every enabled augmentation, in a fixed physiologically-sensible order.

        Order rationale: geometric transforms (crop/stretch/shift/dropout) run
        before amplitude/noise transforms, so noise is added to the final
        waveform shape rather than being distorted by a subsequent resample.
        '''
        out = signal.copy()
        for op_name in self._OP_ORDER:
            if self._roll(op_name):
                out = getattr(self, op_name)(out)
        return out.astype(np.float32)
