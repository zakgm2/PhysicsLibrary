"""
analysis/group_trials.py
------------------------
Trials for group analysis: the window around every event of a marker, baseline-corrected
and reduced to the five measures (what each one is: see group.py). It works on plain
arrays (the time axis, the dF/F, and the event times of each marker), so it does not care
how a recording was loaded.

A trial is *usable* if its whole window, [event - pre, event + post], lies inside the
recording. Every usable trial becomes one row, in time order; a trial that cannot be
measured (a flat baseline for the z-score, non-finite samples, an empty window) stays in
the table with valid = False and the reason, so nothing disappears silently.
"""

import numpy as np
import pandas as pd

from .auc import compute_auc_from_trace
from .shared import estimate_sample_rate, smooth_signal
from .zscore_peth import _baseline_mean_std

MEASURE_COLUMNS = ("auc", "peak", "mean", "latency", "decay")
TRIAL_COLUMNS = ["group", "subject", "recording", "marker", "trial", "event_time", "valid", "reason",
                 *MEASURE_COLUMNS, "baseline_mean", "baseline_sd", "n_samples", "overlap"]
TRACE_STEP = 0.04   # seconds between points of the mean traces (the dF/F is low-passed at 5 Hz, far below this rate)


def decay_time(t, y, peak_index, fraction):
    """
    Seconds from the peak until the trace has fallen to `fraction` of the peak, measured by linear
    interpolation between the two samples either side of the crossing. NaN if the trace never falls that
    far after the peak, or if there is no peak to fall from (a peak of exactly zero).

    A positive peak has fallen when the trace is at or below fraction * peak; a negative peak (a dip)
    when it is at or above it, i.e. when it has climbed back that far.

    Parameters
    ----------
    t, y : arrays
        Time (seconds from the event) and the baseline-corrected trace, the whole trial.
    peak_index : int
        Index of the peak in t / y.
    fraction : float
        Between 0 and 1 (0.5 = half-decay time).
    """
    peak = y[peak_index]
    if not np.isfinite(peak) or peak == 0:
        return float("nan")
    target = fraction * peak
    tail = y[peak_index:]
    hits = np.flatnonzero(tail <= target if peak > 0 else tail >= target)
    if hits.size == 0:
        return float("nan")
    j = int(hits[0])                       # >= 1: the peak itself is beyond the target for any fraction below 1
    t0, t1 = t[peak_index + j - 1], t[peak_index + j]
    y0, y1 = tail[j - 1], tail[j]
    frac = (target - y0) / (y1 - y0) if y1 != y0 else 1.0
    return float(t0 + frac * (t1 - t0) - t[peak_index])


def measures_for_trial(rel_t, y, spec):
    """
    The five measures of one baseline-corrected trial.

    Parameters
    ----------
    rel_t : array
        Time in seconds relative to the event.
    y : array
        The baseline-corrected trace (dF/F minus the baseline mean, or the z-score, per spec.signal).
    spec : GroupSpec
        Uses response, peak_direction and decay_fraction.

    Returns
    -------
    dict with auc, peak, mean, latency, decay — or None if the response window holds fewer than 2 samples.
    """
    r0, r1 = spec.response
    idx = np.flatnonzero((rel_t >= r0) & (rel_t <= r1))
    if idx.size < 2:
        return None
    rt, ry = rel_t[idx], y[idx]
    if spec.peak_direction == "positive":
        k = int(np.argmax(ry))
    elif spec.peak_direction == "negative":
        k = int(np.argmin(ry))
    else:
        k = int(np.argmax(np.abs(ry)))
    return {
        "auc": float(compute_auc_from_trace(rt, ry)["total_auc"]),
        "peak": float(ry[k]),
        "mean": float(ry.mean()),
        "latency": float(rt[k]),
        "decay": decay_time(rel_t, y, int(idx[k]), spec.decay_fraction),
    }


def extract_group_trials(x, y, events, spec, subject, recording="", trace_step=TRACE_STEP):
    """
    Slice one recording into trials and measure each.

    Parameters
    ----------
    x : array
        Time axis of the recording in seconds (uniformly sampled; need not start at 0).
    y : array
        The dF/F, same length as x. Smoothed here with spec.smooth_seconds (0 = not at all).
    events : dict {marker name: sequence of event times}
        Only the markers in spec.markers are used; one with no events gives no trials.
    spec : GroupSpec
        Uses group_name, markers, pre, post, baseline, response, signal, smooth_seconds,
        peak_direction and decay_fraction.
    subject, recording : str
        Filled into every row.
    trace_step : float
        Spacing in seconds of the mean traces returned alongside.

    Returns
    -------
    dict with
        trials   : DataFrame with TRIAL_COLUMNS, one row per usable trial ordered by marker (as in
                   spec.markers) then time. `trial` counts within the marker; `overlap` is True when the
                   window shares samples with the neighbouring trial's (events less than pre + post apart).
        traces   : {marker: {"n": valid trials, "mean": mean baseline-corrected trace on `grid`}}
        grid     : the common time axis of the traces, -pre to +post
        excluded : {marker: events whose window is not fully inside the recording}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fs = estimate_sample_rate(x)
    ys = smooth_signal(y, fs, spec.smooth_seconds) if spec.smooth_seconds > 0 else y
    pre, post = spec.pre, spec.post
    b0, b1 = spec.baseline
    grid = np.arange(-pre, post + trace_step / 2, trace_step)

    rows, traces, excluded = [], {}, {}
    for marker in spec.markers:
        times = np.sort(np.asarray(events.get(marker, []), dtype=float))
        inside = (times - pre >= x[0]) & (times + post <= x[-1])
        usable = times[inside]
        excluded[marker] = int(len(times) - len(usable))

        overlap = np.zeros(len(usable), dtype=bool)
        if len(usable) > 1:
            close = np.diff(usable) < pre + post
            overlap[:-1] |= close
            overlap[1:] |= close

        total, n_valid = np.zeros(len(grid)), 0
        for number, t in enumerate(usable, start=1):
            i0 = int(np.searchsorted(x, t - pre, side="left"))
            i1 = int(np.searchsorted(x, t + post, side="right"))
            rel, seg = x[i0:i1] - t, ys[i0:i1]
            row = {"group": spec.group_name, "subject": subject, "recording": recording, "marker": marker,
                   "trial": number, "event_time": float(t), "valid": False, "reason": "",
                   "baseline_mean": np.nan, "baseline_sd": np.nan, "n_samples": int(len(seg)),
                   "overlap": bool(overlap[number - 1]), **{c: np.nan for c in MEASURE_COLUMNS}}
            base = seg[(rel >= b0) & (rel <= b1)]
            if base.size == 0:
                row["reason"] = "no samples in the baseline window"
            elif not np.isfinite(seg).all():
                row["reason"] = "non-finite samples in the window"
            else:
                mu, sd = _baseline_mean_std(base)
                row["baseline_mean"], row["baseline_sd"] = float(mu), float(sd)
                corrected = None
                if spec.signal == "zscore":
                    if sd == 0 or sd <= 1e-12 * np.max(np.abs(base)):
                        row["reason"] = "flat baseline (no spread to z-score by)"
                    else:
                        corrected = (seg - mu) / sd
                else:
                    corrected = seg - mu
                if corrected is not None:
                    measured = measures_for_trial(rel, corrected, spec)
                    if measured is None:
                        row["reason"] = "fewer than 2 samples in the response window"
                    else:
                        row.update(measured)
                        row["valid"] = True
                        total += np.interp(grid, rel, corrected)
                        n_valid += 1
            rows.append(row)
        traces[marker] = {"n": n_valid, "mean": total / n_valid if n_valid else np.full(len(grid), np.nan)}

    trials = pd.DataFrame(rows, columns=TRIAL_COLUMNS)
    return {"trials": trials, "traces": traces, "grid": grid, "excluded": excluded}
