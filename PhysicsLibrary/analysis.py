"""
analysis.py
-----------
Format-agnostic analysis routines for Physics Analysis GUI.

Currently:
  - Z-Score PETH (get_zscore_slice, smooth_signal, bin_for_heatmap)

Working on: 
  - Fourier Transform 
"""

import numpy as np


def get_zscore_slice(time_array, signal, center_t, window=30):
    """
    Extract and z-score a time window around an event.

    Parameters
    ----------
    time_array : array
    signal : array
    center_t : float
        Event time in seconds
    window : float
        Total window size in seconds

    Returns
    -------
    (time segment, z-scored signal)
    """
    half_win  = window / 2
    start_idx = np.searchsorted(time_array, center_t - half_win)
    end_idx   = np.searchsorted(time_array, center_t + half_win)

    seg_y = signal[start_idx:end_idx]
    seg_x = time_array[start_idx:end_idx]

    seg_y = np.clip(seg_y, -5, 5)

    baseline_end    = len(seg_y) // 2
    baseline_period = seg_y[:baseline_end]
    mu  = np.mean(baseline_period)
    std = np.std(baseline_period)

    if std < 1e-6:
        return seg_x, np.zeros_like(seg_y)

    return seg_x, (seg_y - mu) / std


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


def bin_for_heatmap(z_seg, num_bins=300):
    """
    Bin a signal into equal segments for heatmap plotting.

    Parameters
    ----------
    z_seg : array
    num_bins : int

    Returns
    -------
    array
        Binned signal
    """
    if z_seg is None or len(z_seg) == 0:
        return np.zeros(num_bins)
    bin_edges = np.linspace(0, len(z_seg), num_bins + 1).astype(int)
    return np.array([np.mean(z_seg[bin_edges[i]:bin_edges[i+1]]) for i in range(num_bins)])