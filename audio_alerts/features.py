"""Level-normalized spectral features from 5-second WAV clips."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly, stft


def extract(path: Path) -> np.ndarray:
    rate, samples = wavfile.read(path)
    if samples.ndim == 2:
        samples = samples.astype(np.float64).mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float64) / max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
    else:
        samples = samples.astype(np.float64)
    if rate != 16000:
        from math import gcd
        divisor = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor)
    if len(samples) < 16000:
        raise ValueError(f"Clip shorter than one second: {path}")
    samples = samples[:16000 * 5]
    samples -= samples.mean()
    samples /= max(np.sqrt(np.mean(samples**2)), 1e-8)
    frequencies, _, spectrum = stft(samples, fs=16000, nperseg=512, noverlap=256,
                                    boundary=None, padded=False)
    power = np.abs(spectrum) ** 2 + 1e-12
    total = power.sum(axis=0)
    centroid = (frequencies[:, None] * power).sum(axis=0) / total
    flatness = np.exp(np.log(power).mean(axis=0)) / power.mean(axis=0)
    band_edges = (0, 250, 500, 1000, 2000, 3000, 4000, 6000, 8001)
    bands = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(power[mask].sum(axis=0) / total)
    frame_features = np.vstack([centroid / 8000, flatness, *bands])
    # Log-mel cepstral coefficients retain timbre information that a centroid
    # alone misses. Build 26 triangular filters at the resampled 16 kHz rate.
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    mel_to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(40), hz_to_mel(8000), 28))
    filters = np.zeros((26, len(frequencies)))
    for index in range(26):
        left, middle, right = edges[index:index + 3]
        filters[index] = np.maximum(0, np.minimum(
            (frequencies - left) / (middle - left),
            (right - frequencies) / (right - middle)))
    mel_power = filters @ power
    mfcc = dct(np.log(mel_power + 1e-10), axis=0, norm="ortho", type=2)[1:13]
    combined = np.vstack([frame_features, mfcc])
    return np.concatenate([combined.mean(axis=1), combined.std(axis=1)])
