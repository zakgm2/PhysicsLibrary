"""
analysis/auc.py
----------------
Area under curve, split by sign. compute_auc_from_trace is the shared
core both compute_auc_window (single window, this file) and
event_peth.py's compute_auc_matrix (per-trial/group, across a whole
trial x time matrix) delegate to, so the sign-split logic lives in
exactly one place.
"""

import numpy as np


def compute_auc_from_trace(x, y):
    """
    Trapezoidal area under an already-extracted trace, split by sign.

    positive_auc integrates only where y >= 0 (negative samples clipped
    to 0); negative_auc integrates only where y <= 0 (positive samples
    clipped to 0) and is a real *signed* area (<= 0), not an absolute
    value — total_auc is their sum, equivalent to the plain net signed
    integral of y.

    Parameters
    ----------
    x : array
    y : array

    Returns
    -------
    dict with total_auc, positive_auc, negative_auc (all float). All
    zero if x has fewer than 2 points (np.trapezoid needs at least 2 to
    integrate anything).
    """
    if len(x) < 2:
        return {"total_auc": 0.0, "positive_auc": 0.0, "negative_auc": 0.0}
    total = float(np.trapezoid(y, x))
    positive = float(np.trapezoid(np.clip(y, 0, None), x))
    negative = float(np.trapezoid(np.clip(y, None, 0), x))
    return {"total_auc": total, "positive_auc": positive, "negative_auc": negative}


def compute_auc_window(time_array, signal, center_t, pre, post):
    """
    Slice a pre/post window around an event and integrate it — same
    windowing convention as get_zscore_slice/compute_fft_slice, so this
    drops into the same click-triggered dispatch path those use.
    Deliberately does NOT z-score the segment first: AUC is meant to
    quantify the signal (already dF/F-normalized upstream) in its own
    units, not a re-baselined version of it.

    Parameters
    ----------
    time_array : array
    signal : array
    center_t : float
        Event time in seconds.
    pre, post : float
        Seconds before/after center_t to include.

    Returns
    -------
    dict with seg_x, seg_y (the windowed slice actually integrated) plus
    total_auc, positive_auc, negative_auc (see compute_auc_from_trace).
    """
    start_idx = np.searchsorted(time_array, center_t - pre)
    end_idx = np.searchsorted(time_array, center_t + post)
    seg_x = time_array[start_idx:end_idx]
    seg_y = signal[start_idx:end_idx]
    result = compute_auc_from_trace(seg_x, seg_y)
    result["seg_x"] = seg_x
    result["seg_y"] = seg_y
    return result
