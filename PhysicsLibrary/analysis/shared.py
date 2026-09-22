"""
analysis/shared.py
-------------------
Small helpers used by more than one analysis tool (or by a loader
outside analysis/ entirely, e.g. estimate_sample_rate for a generic
tabular file with no reported fs) — kept separate from any one tool's
own file so pulling in, say, curve_fit.py doesn't also drag in AUC- or
FFT-specific code that happens to need the same helper.
"""

import numpy as np


def estimate_sample_rate(time_array):
    """
    Robust sample-rate estimate from a timestamp array, for a source that
    doesn't report its own fs (e.g. a generic tabular file) or as a
    fallback wherever one's needed internally. Uses the median inter-
    sample interval rather than e.g. `1 / (time_array[1] - time_array[0])`
    so one irregular gap (a dropped sample, a rounding artifact at the
    very start) doesn't skew the whole estimate.

    Parameters
    ----------
    time_array : array
        Must have at least 2 samples.

    Returns
    -------
    float, sample rate in Hz.
    """
    return float(1.0 / np.median(np.diff(time_array)))


def mean_channels(arr):
    """
    Collapse a multi-channel array to its across-channel mean trace —
    e.g. Oxysoft's per-detector o2hb/hhb/thb arrays, shape
    (n_channels, n_samples), averaged down to one representative trace.
    A no-op for an already-1D array, so callers can use this unconditionally
    regardless of whether a given recording actually has multiple channels.

    Parameters
    ----------
    arr : array, 1D or 2D (n_channels, n_samples)

    Returns
    -------
    array, 1D — arr unchanged if already 1D, else arr.mean(axis=0).
    """
    return arr.mean(axis=0) if arr.ndim > 1 else arr


def smooth_signal(data, fs, window_sec=0.5):
    """
    Moving average smoothing filter.

    Parameters
    ----------
    data : array
    fs : float
        Sampling frequency in Hz
    window_sec : float
        Smoothing window in seconds

    Returns
    -------
    array
        Smoothed signal
    """
    window_size = int(fs * window_sec)
    if window_size % 2 == 0:
        window_size += 1
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')
