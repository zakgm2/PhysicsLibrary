"""
analysis/zscore_peth.py
------------------------
Single-click Z-Score PETH: extract and z-score a window around one
event. get_zscore_slice is also the shared building block event_peth.py
and peak_finder.py call per-event/per-trial, so a bug fix or behavior
change here (e.g. the artefact clip below) applies everywhere a peri-
event window gets z-scored, not just this one tool.
"""

import numpy as np


def get_zscore_slice(time_array, signal, center_t, window=None, pre=None, post=None):
    """
    Extract and z-score a time window around an event.

    Supports either a symmetric `window` (split evenly before/after
    center_t, legacy behaviour) or an asymmetric `pre`/`post` pair — e.g.
    10s before the event and 20s after. If pre/post are given they take
    precedence over window.

    Parameters
    ----------
    time_array : array
    signal : array
    center_t : float
        Event time in seconds
    window : float or None
        Total window size in seconds, split evenly before/after center_t.
        Ignored if pre/post are given.
    pre : float or None
        Seconds before center_t to include. Defaults to window/2.
    post : float or None
        Seconds after center_t to include. Defaults to window/2.

    Returns
    -------
    (time segment, z-scored signal)
    """
    if pre is None or post is None:
        half_win = (window if window is not None else 30) / 2
        pre  = pre  if pre  is not None else half_win
        post = post if post is not None else half_win

    start_idx = np.searchsorted(time_array, center_t - pre)
    end_idx   = np.searchsorted(time_array, center_t + post)

    seg_y = signal[start_idx:end_idx]
    seg_x = time_array[start_idx:end_idx]

    # Clip extreme artefacts before z-scoring so outliers don't dominate the baseline std.
    seg_y = np.clip(seg_y, -5, 5)

    # Baseline is the pre-event portion — the part of the window that
    # actually precedes the event, not just "the first half of the segment"
    # (those differ once pre != post).
    baseline_mask   = seg_x < center_t
    baseline_period = seg_y[baseline_mask] if baseline_mask.any() else seg_y
    mu  = np.mean(baseline_period)
    std = np.std(baseline_period)

    if std < 1e-6:
        return seg_x, np.zeros_like(seg_y)

    return seg_x, (seg_y - mu) / std


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
