"""
analysis/event_peth.py
------------------------
Stacked-heatmap + trial-average PETH (GuPPy-style): align every
occurrence of one event type into a trial x time matrix, then group
stats (mean/SEM), AUC, and peak/latency/mean-time-bins over that matrix
or an arbitrary subset of its rows (e.g. Event PETH's currently-checked
trials). compute_auc_matrix lives here rather than in auc.py because its
whole shape (per_trial/group over a trial_matrix) only exists in this
stacked-trials context — it delegates to auc.py's compute_auc_from_trace
for the actual per-row integration rather than duplicating it.
"""

import numpy as np

from .zscore_peth import get_zscore_slice
from .auc import compute_auc_from_trace


def compute_group_stats(trial_matrix):
    """
    Mean + SEM across trials (rows) of an already-built trial x time
    matrix (see compute_event_zscore_peth) — the same formula that
    function uses internally for its own full-trial-set mean_trace/
    sem_trace, exposed separately so a caller needing it for an arbitrary
    subset of trials (e.g. Event PETH's currently-checked trials, not
    necessarily every trial compute_event_zscore_peth originally computed)
    doesn't have to re-derive the formula.

    Parameters
    ----------
    trial_matrix : array, shape (n_trials, n_samples)

    Returns
    -------
    (mean_trace, sem_trace) : both array, shape (n_samples,). sem_trace
        is all zeros if trial_matrix has fewer than 2 rows (ddof=1 SEM is
        undefined below 2 samples) — matches compute_event_zscore_peth's
        own zero-trials/one-trial handling. mean_trace is all zeros for
        0 rows (nothing to average).
    """
    n = trial_matrix.shape[0]
    if n == 0:
        return np.zeros(trial_matrix.shape[1]), np.zeros(trial_matrix.shape[1])
    mean_trace = trial_matrix.mean(axis=0)
    if n > 1:
        sem_trace = trial_matrix.std(axis=0, ddof=1) / np.sqrt(n)
    else:
        sem_trace = np.zeros_like(mean_trace)
    return mean_trace, sem_trace


def compute_event_zscore_peth(time_array, signal, event_times, pre, post, num_bins=300):
    """
    Z-score and align every occurrence of one event type into a
    trial x time matrix, for a stacked-heatmap + trial-average PETH view
    (GuPPy-style) rather than a single click-triggered PETH.

    Each trial is z-scored independently against its own pre-event
    baseline (see get_zscore_slice) — that's what makes a trial's
    response comparable regardless of the signal's absolute level at
    that point in the recording. Trials are then resampled onto one
    shared relative-time axis (num_bins points spanning -pre..+post) so
    they can be stacked into a single matrix despite each trial's raw
    segment having a slightly different sample count from indexing
    rounding.

    Parameters
    ----------
    time_array : array
    signal : array
    event_times : list of float
        Timestamps (same units as time_array) for every occurrence of
        the event being analyzed.
    pre, post : float
        Seconds before/after each event to include.
    num_bins : int
        Number of points each trial is resampled to.

    Returns
    -------
    dict with:
        time_axis : array, shape (num_bins,) — relative time, -pre..+post
        trial_matrix : array, shape (n_valid_trials, num_bins)
        trial_event_times : list of the event_times that produced a
            usable trial (too-short/edge-of-recording events are skipped)
        mean_trace : array, shape (num_bins,)
        sem_trace : array, shape (num_bins,) — standard error of the mean
            across trials, zero if fewer than 2 trials
    """
    time_axis = np.linspace(-pre, post, num_bins)
    rows = []
    valid_times = []
    for t in event_times:
        seg_x, seg_z = get_zscore_slice(time_array, signal, t, pre=pre, post=post)
        if seg_x is None or len(seg_x) < 2:
            continue
        rel_x = seg_x - t
        rows.append(np.interp(time_axis, rel_x, seg_z))
        valid_times.append(t)

    if not rows:
        empty = np.zeros(num_bins)
        return {
            "time_axis": time_axis, "trial_matrix": np.zeros((0, num_bins)),
            "trial_event_times": [], "mean_trace": empty, "sem_trace": empty,
        }

    trial_matrix = np.array(rows)
    mean_trace, sem_trace = compute_group_stats(trial_matrix)

    return {
        "time_axis": time_axis, "trial_matrix": trial_matrix,
        "trial_event_times": valid_times, "mean_trace": mean_trace, "sem_trace": sem_trace,
    }


def compute_auc_matrix(time_axis, trial_matrix):
    """
    Per-trial + group (mean-trace) AUC for an already-built trial x time
    matrix (see compute_event_zscore_peth) — lets Event PETH compute its
    group stat and per-trial list in one call over whichever subset of
    trials is currently relevant, without re-deriving the trapezoidal/
    clip logic per caller.

    Parameters
    ----------
    time_axis : array, shape (num_bins,)
    trial_matrix : array, shape (n_trials, num_bins)

    Returns
    -------
    dict with:
        per_trial : list of dict (one per row of trial_matrix, same
            order), each {"total_auc", "positive_auc", "negative_auc"}
        group : dict, same three keys, computed on trial_matrix's
            mean(axis=0) — all zero if trial_matrix has 0 rows
    """
    per_trial = [compute_auc_from_trace(time_axis, row) for row in trial_matrix]
    if trial_matrix.shape[0] > 0:
        group = compute_auc_from_trace(time_axis, trial_matrix.mean(axis=0))
    else:
        group = {"total_auc": 0.0, "positive_auc": 0.0, "negative_auc": 0.0}
    return {"per_trial": per_trial, "group": group}


def _mean_time_bins(rel_x, y, post, bin_width):
    """Post-event window [0, post] divided into bin_width-second chunks
    (the last bin is truncated, not dropped, if post isn't evenly
    divisible by bin_width). rel_x is already relative to the event
    (0 = event time).

    Returns
    -------
    list of dict, each {"bin_start", "bin_end", "mean"} — empty if
    post <= 0, bin_width <= 0, or there are no rel_x >= 0 samples.
    """
    if post <= 0 or bin_width <= 0:
        return []
    bins = []
    edge = 0.0
    while edge < post:
        bin_end = min(edge + bin_width, post)
        mask = (rel_x >= edge) & (rel_x < bin_end)
        mean_val = float(np.mean(y[mask])) if mask.any() else float("nan")
        bins.append({"bin_start": edge, "bin_end": bin_end, "mean": mean_val})
        edge = bin_end
    return bins


def compute_peri_event_from_trace(rel_x, y, post, bin_width=1.0):
    """
    Peak amplitude, time-to-peak (latency), and mean-time-bins from an
    already-extracted, already-relative-to-event trace (0 = event time).

    Peak amplitude/latency are deliberately post-event only (rel_x >= 0)
    — a pre-event sample can't be a response to the event. Uses the same
    signed argmax(|y|) convention peth.py's own single-click peak-Z stat
    already uses, rather than a plain positive-only max, so "peak" can
    correctly capture a strong inhibitory (negative-going) response too.

    Parameters
    ----------
    rel_x : array
        Time relative to the event (seconds), 0 = event time.
    y : array
        Signal values aligned to rel_x.
    post : float
        Seconds after the event covered by rel_x/y — bounds the
        mean-time-bins range and is passed separately (rather than
        inferred from rel_x.max()) so a caller with a fixed window size
        (e.g. Event PETH, same post for every trial) gets identical bin
        edges across trials even if a particular trial's samples don't
        quite reach the nominal edge.
    bin_width : float
        Width of each mean-time-bin, in seconds.

    Returns
    -------
    dict with peak_amplitude, latency (both float — 0.0 if no post-event
    samples exist) and mean_bins (list of dict, see _mean_time_bins).
    """
    mask = rel_x >= 0
    if not mask.any():
        peak_amplitude, latency = 0.0, 0.0
    else:
        rx, ry = rel_x[mask], y[mask]
        idx = int(np.argmax(np.abs(ry)))
        peak_amplitude, latency = float(ry[idx]), float(rx[idx])
    return {
        "peak_amplitude": peak_amplitude,
        "latency": latency,
        "mean_bins": _mean_time_bins(rel_x, y, post, bin_width),
    }


def compute_peri_event_matrix(time_axis, trial_matrix, post, bin_width=1.0):
    """
    Per-trial + group (mean-trace) peak amplitude / time-to-peak /
    mean-time-bins for an already-built trial x time matrix (see
    compute_event_zscore_peth) — mirrors compute_auc_matrix exactly, so
    Event PETH can compute both AUC and these stats the same way.

    Parameters
    ----------
    time_axis : array, shape (num_bins,)
        Already relative to the event (-pre..+post), per
        compute_event_zscore_peth's convention.
    trial_matrix : array, shape (n_trials, num_bins)
    post : float
        Seconds after the event covered by time_axis.
    bin_width : float
        Width of each mean-time-bin, in seconds.

    Returns
    -------
    dict with:
        per_trial : list of dict (one per row of trial_matrix, same
            order), each {"peak_amplitude", "latency", "mean_bins"}
        group : dict, same shape, computed on trial_matrix's
            mean(axis=0) — peak/latency 0.0 and mean_bins empty if
            trial_matrix has 0 rows
    """
    per_trial = [compute_peri_event_from_trace(time_axis, row, post, bin_width)
                 for row in trial_matrix]
    if trial_matrix.shape[0] > 0:
        group = compute_peri_event_from_trace(time_axis, trial_matrix.mean(axis=0), post, bin_width)
    else:
        group = {"peak_amplitude": 0.0, "latency": 0.0, "mean_bins": []}
    return {"per_trial": per_trial, "group": group}
