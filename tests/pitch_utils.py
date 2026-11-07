"""A small, independent pitch (f0) estimator used to check the synth.

It's deliberately *not* derived from the synth's own math: if the synth were
wrong about its pitch, an estimator that shared its assumptions could be wrong
in the same way and the test would still pass.

Two stages:
1. Coarse, octave-safe guess with YIN (de Cheveigne & Kawahara, 2002).
   Plain "biggest spectral peak" often picks the 2nd or 3rd harmonic of a
   bright note (an octave error). YIN instead asks "after how many samples does
   the waveform best repeat itself?" and prefers the *shortest* such lag, which
   is the true period.
2. Fine estimate from the spectrum: around the coarse guess, find the peak of a
   long, zero-padded FFT and interpolate between bins with a parabola. A 1 s
   window resolves ~1 Hz; interpolation gets well below that, which we need:
   10 cents at 82 Hz is only ~0.5 Hz.
"""

import numpy as np


def yin_period(x: np.ndarray, min_lag: int, max_lag: int, threshold: float = 0.15) -> int:
    window = len(x) - max_lag
    # d(tau) = sum_j (x[j] - x[j+tau])^2: how different the signal is from
    # itself shifted by tau. Zero at a perfect repeat.
    d = np.array(
        [np.sum((x[:window] - x[tau : tau + window]) ** 2) for tau in range(max_lag + 1)]
    )
    # Cumulative-mean normalization: divide by the running average so small
    # lags (where d is trivially small) don't win, and a fixed threshold works
    # regardless of loudness.
    cmnd = np.ones_like(d)
    cumsum = np.cumsum(d[1:])
    cmnd[1:] = d[1:] * np.arange(1, max_lag + 1) / np.maximum(cumsum, 1e-12)

    for tau in range(min_lag, max_lag):
        if cmnd[tau] < threshold:
            while tau + 1 < max_lag and cmnd[tau + 1] < cmnd[tau]:
                tau += 1  # slide down to the bottom of this dip
            return tau
    return int(np.argmin(cmnd[min_lag:max_lag]) + min_lag)


def estimate_f0(
    audio: np.ndarray, sample_rate: int, fmin: float = 60.0, fmax: float = 1500.0
) -> float:
    # Skip the noisy first 50 ms (the pluck itself), analyze up to 1 s.
    seg = np.asarray(audio, dtype=np.float64)[int(0.05 * sample_rate) : int(1.05 * sample_rate)]
    seg = seg - seg.mean()

    min_lag = int(sample_rate / fmax)
    max_lag = int(np.ceil(sample_rate / fmin))
    coarse_seg = seg[: 2048 + max_lag]
    coarse_hz = sample_rate / yin_period(coarse_seg, min_lag, max_lag)

    n_fft = 1 << 18  # heavy zero-padding = finely sampled spectrum
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    band = np.where((freqs > coarse_hz * 0.95) & (freqs < coarse_hz * 1.05))[0]
    k = band[np.argmax(spectrum[band])]

    # Parabolic interpolation on log magnitude through bins k-1, k, k+1.
    a, b, c = np.log(spectrum[k - 1 : k + 2] + 1e-20)
    offset = 0.5 * (a - c) / (a - 2 * b + c)
    return float((k + offset) * sample_rate / n_fft)
