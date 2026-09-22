"""
analysis/peak_finder.py
-------------------------
Statistically significant transients: either auto-detected straight
from the signal (find_significant_peaks), or checked for near a given
set of event times (find_peak_near_events) rather than assuming an
event marker itself marks where the neural signal responds.
"""

import numpy as np
from scipy.signal import find_peaks

from .shared import estimate_sample_rate
from .zscore_peth import get_zscore_slice


def find_significant_peaks(time_array, signal, z_threshold=2.5, min_distance_sec=1.0,
                            include_troughs=False):
    """
    Auto-detect statistically significant transients directly from the
    signal, rather than relying on externally-supplied event markers
    (TDT epocs, manual markers, ...) that may not actually line up with
    where the neural signal itself is doing something.

    The whole recording is z-scored against its own global mean/std
    (not a local baseline — this is a single-pass "how unusual is this
    point relative to the entire recording" measure, not per-event), and
    scipy.signal.find_peaks picks local maxima at or above z_threshold,
    at least min_distance_sec apart so a single transient's rising edge
    doesn't get counted as several peaks.

    Parameters
    ----------
    time_array : array
    signal : array
        Already-processed signal (e.g. bleach-corrected + smoothed) —
        this function does no filtering of its own.
    z_threshold : float
        Minimum z-score (standard deviations above the recording's own
        mean) for a peak to count as "statistically significant".
    min_distance_sec : float
        Minimum spacing between detected peaks, in seconds.
    include_troughs : bool
        Also detect significant negative-going deflections (z <=
        -z_threshold) — off by default since most fibre-photometry
        analyses care about excitatory transients specifically.

    Returns
    -------
    list of dict, each {"time": float, "z_score": float, "kind": "peak"|"trough"},
    sorted by time.
    """
    fs = estimate_sample_rate(time_array)
    distance = max(1, int(min_distance_sec * fs))

    mu, std = np.mean(signal), np.std(signal)
    if std < 1e-9:
        return []
    z = (signal - mu) / std

    results = []
    peak_idx, _ = find_peaks(z, height=z_threshold, distance=distance)
    for i in peak_idx:
        results.append({"time": float(time_array[i]), "z_score": float(z[i]), "kind": "peak"})

    if include_troughs:
        trough_idx, _ = find_peaks(-z, height=z_threshold, distance=distance)
        for i in trough_idx:
            results.append({"time": float(time_array[i]), "z_score": float(z[i]), "kind": "trough"})

    results.sort(key=lambda r: r["time"])
    return results


def find_peak_near_events(time_array, signal, event_times, pre, post,
                           z_threshold=2.5, include_troughs=False):
    """
    Check whether a statistically significant peak actually shows up near
    each given event time, rather than assuming the event marker itself
    marks where the neural signal responds. Works for a single event
    (event_times of length 1) or many occurrences of the same event type
    (checking consistency across all of them).

    Each event's window is baselined the same way as get_zscore_slice
    (pre-event portion), so "significant" means relative to that event's
    own local baseline, not the whole recording's.

    Parameters
    ----------
    time_array : array
    signal : array
    event_times : list of float
    pre, post : float
        Seconds before/after each event to search within.
    z_threshold : float
        Minimum |z-score| within the window for a peak to count as found.
    include_troughs : bool
        Also consider negative-going deflections as candidate "peaks",
        keeping whichever (peak or trough) is more extreme.

    Returns
    -------
    list of dict, one per event_time (same order), each:
        {"event_time": float, "found": bool, "peak_time": float or None,
         "latency": float or None (peak_time - event_time),
         "z_score": float or None, "kind": "peak"|"trough"|None}
    "found" is False when the window was unusable (too close to the
    recording's edges) or nothing in it reached z_threshold.
    """
    results = []
    for t in event_times:
        seg_x, seg_z = get_zscore_slice(time_array, signal, t, pre=pre, post=post)
        if seg_x is None or len(seg_x) == 0:
            results.append({"event_time": t, "found": False, "peak_time": None,
                             "latency": None, "z_score": None, "kind": None})
            continue

        idx_max = int(np.argmax(seg_z))
        if include_troughs:
            idx_min = int(np.argmin(seg_z))
            if abs(seg_z[idx_min]) > seg_z[idx_max]:
                best_idx, kind = idx_min, "trough"
            else:
                best_idx, kind = idx_max, "peak"
        else:
            best_idx, kind = idx_max, "peak"

        best_z = float(seg_z[best_idx])
        found = abs(best_z) >= z_threshold
        peak_time = float(seg_x[best_idx]) if found else None
        results.append({
            "event_time": t, "found": found,
            "peak_time": peak_time,
            "latency": (peak_time - t) if found else None,
            "z_score": best_z if found else None,
            "kind": kind if found else None,
        })
    return results
