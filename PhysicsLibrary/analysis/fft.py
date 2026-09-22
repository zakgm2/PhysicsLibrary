"""
analysis/fft.py
-----------------
FFT of a windowed signal segment, plus picking out its top peaks as
data. Drawing those peaks onto a plot (labels, arrows, a vertical line)
is presentation, not computation, and lives in physicsanalysis_qt's own
analysis/fft.py instead — this file only ever returns numbers.
"""

import numpy as np
from scipy.signal import detrend, find_peaks


def compute_fft_slice(time_array, signal, center_t, fs, window=None, pre=None, post=None):
    """
    Extract a time window around center_t and compute its FFT.

    Applies mean removal and linear detrending before FFT to eliminate
    the DC spike and slow drift, making physiological frequencies
    (breathing ~0.3 Hz, heart rate ~1 Hz) visible.

    Supports either a symmetric `window` (split evenly before/after
    center_t, legacy behaviour) or an asymmetric `pre`/`post` pair. If
    pre/post are given they take precedence over window.

    Parameters
    ----------
    time_array : array
    signal : array
    center_t : float
        Center time in seconds
    fs : float
        Sampling frequency in Hz
    window : float or None
        Total window size in seconds, split evenly before/after center_t.
        Ignored if pre/post are given.
    pre : float or None
        Seconds before center_t to include. Defaults to window/2.
    post : float or None
        Seconds after center_t to include. Defaults to window/2.

    Returns
    -------
    freqs : array
    power : array
    seg_x : array
    seg_y : array
    """
    if pre is None or post is None:
        half_win = (window if window is not None else 30) / 2
        pre  = pre  if pre  is not None else half_win
        post = post if post is not None else half_win

    start_idx = np.searchsorted(time_array, center_t - pre)
    end_idx   = np.searchsorted(time_array, center_t + post)

    seg_y = signal[start_idx:end_idx]
    seg_x = time_array[start_idx:end_idx]

    if len(seg_y) < 4:
        return np.array([]), np.array([]), seg_x, seg_y

    seg_y    = detrend(seg_y, type='linear')   # removes mean and linear trend
    windowed = seg_y * np.hanning(len(seg_y))

    n     = len(windowed)
    fft_y = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = (np.abs(fft_y) ** 2) / n

    return freqs, power, seg_x, seg_y


def find_fft_peaks(freqs, power, n_peaks=3):
    """
    Find the top N peaks in a power spectrum, as data.

    Parameters
    ----------
    freqs  : array
    power  : array
    n_peaks: int

    Returns
    -------
    list of dict, each {"freq_hz": float, "power": float, "bpm": float},
    sorted by power descending. Empty if fewer than 3 usable frequency
    bins (freqs >= 0.05 Hz) exist, or no peak clears the prominence bar.
    """
    mask     = freqs >= 0.05
    f_m      = freqs[mask]
    p_m      = power[mask]
    if len(p_m) < 3:
        return []
    min_prom = 0.05 * p_m.max()
    peaks, _ = find_peaks(p_m, prominence=min_prom)
    if len(peaks) == 0:
        return []
    top = sorted(peaks, key=lambda i: p_m[i], reverse=True)[:n_peaks]
    return [
        {"freq_hz": float(f_m[i]), "power": float(p_m[i]), "bpm": float(f_m[i] * 60)}
        for i in top
    ]
