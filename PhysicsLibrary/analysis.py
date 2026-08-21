"""
analysis.py
-----------
Format-agnostic analysis routines for Physics Analysis GUI.

Currently:
  - Z-Score PETH (get_zscore_slice, smooth_signal, bin_for_heatmap)
  - FFT (compute_fft_slice, find_fft_peaks, annotate_fft_peaks)
  - Curve fitting (compute_slope_segment, fit_model_to_segment)
  - Area under curve (compute_auc_from_trace, compute_auc_window, compute_auc_matrix)
  - Peri-event stats: peak amplitude, time-to-peak, mean time bins
    (compute_peri_event_from_trace, compute_peri_event_matrix) — used by
    Event PETH's per-trial/group stats
  - Shared small helpers (estimate_sample_rate, mean_channels,
    compute_group_stats, compute_marker_intervals)
"""

import numpy as np
from scipy.signal import detrend, find_peaks
from scipy.optimize import curve_fit


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


def compute_marker_intervals(markers):
    """
    Time-since-previous-marker, two ways, for a list of already-placed
    markers — universal to any marker source (TDT epocs, Oxysoft events,
    manually-placed), since it only looks at each marker's own 'time'/
    'store'/'label'/'phase' keys, not any format-specific structure.

    For a store with alternating high/low phases (e.g. a pump or lever),
    a low marker's dt_store IS its on-duration (time since its own high).
    For a store with no phase concept, dt_store is just the interval
    since that store's last event. dt_global is the interval since
    whatever marker came before it, regardless of store — useful when
    several event types are interleaved. No special-casing for phase is
    needed — both columns are computed the same generic way (diff
    against the previous timestamp in the relevant sequence), and
    duration falls out of that automatically wherever high/low pairs
    happen to alternate.

    Parameters
    ----------
    markers : list of dict, each with at least 'time' and 'label' keys,
        optionally 'store' (falls back to 'label' if absent) and 'phase'.

    Returns
    -------
    list of dict, sorted by time, each {"time", "store", "label",
    "phase", "dt_store", "dt_global"} — dt_* are None for the first
    event in their respective sequence.
    """
    ordered = sorted(markers, key=lambda m: m['time'])
    last_time_by_store = {}
    rows = []
    prev_time = None
    for m in ordered:
        store = m.get('store') or m['label']
        dt_store = None
        if store in last_time_by_store:
            dt_store = m['time'] - last_time_by_store[store]
        last_time_by_store[store] = m['time']

        dt_global = None if prev_time is None else m['time'] - prev_time
        prev_time = m['time']

        rows.append({
            'time': m['time'],
            'store': store,
            'label': m['label'],
            'phase': m.get('phase', ''),
            'dt_store': dt_store,
            'dt_global': dt_global,
        })
    return rows


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
    Find the top N peaks in a power spectrum, as data rather than a
    drawn annotation — factored out of annotate_fft_peaks (which now
    calls this and only handles the drawing) so a caller building a
    stats/export panel can get the same peak list annotate_fft_peaks
    would have drawn, without re-deriving the selection logic.

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


def annotate_fft_peaks(ax_f, freqs, power, color, n_peaks=3):
    """
    Find top N peaks in a power spectrum and annotate them with
    frequency and BPM labels directly on the axes.

    Parameters
    ----------
    ax_f   : matplotlib Axes
    freqs  : array
    power  : array
    color  : str
    n_peaks: int
    """
    for p in find_fft_peaks(freqs, power, n_peaks=n_peaks):
        ax_f.annotate(
            f"{p['freq_hz']:.2f} Hz\n({p['bpm']:.0f} bpm)",
            xy=(p['freq_hz'], p['power']),
            xytext=(p['freq_hz'] + 0.05, p['power'] * 0.92),
            fontsize=7, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
        )
        ax_f.axvline(p['freq_hz'], color=color, lw=0.7, linestyle=':', alpha=0.5)


def compute_slope_segment(x_data, y_data, p1_idx, p2_idx, padding_pct=0.05):
    """
    Least-squares linear regression slope between two index boundaries.

    Parameters
    ----------
    x_data      : array
    y_data      : array
    p1_idx      : int
    p2_idx      : int
    padding_pct : float   visual context padding

    Returns
    -------
    dict with slope, intercept, crop_x, crop_y, x1, y1, x2, y2
    """
    idx1, idx2 = sorted([p1_idx, p2_idx])

    fit_x = x_data[idx1:idx2 + 1]
    fit_y = y_data[idx1:idx2 + 1]

    if len(fit_x) < 2:
        slope, intercept = 0.0, 0.0
    else:
        slope, intercept = np.polyfit(fit_x, fit_y, 1)

    x1, y1 = fit_x[0],  fit_y[0]
    x2, y2 = fit_x[-1], fit_y[-1]

    pad        = max(5, int(len(x_data) * padding_pct))
    start_idx  = max(0, idx1 - pad)
    end_idx    = min(len(x_data) - 1, idx2 + pad)

    return {
        'slope':     slope,
        'intercept': intercept,
        'crop_x':    x_data[start_idx:end_idx + 1],
        'crop_y':    y_data[start_idx:end_idx + 1],
        'x1': x1, 'y1': y1,
        'x2': x2, 'y2': y2,
    }


def fit_model_to_segment(x_seg, y_seg, model_fn, p0_fn):
    """
    Fit a model function to a data segment using scipy curve_fit.

    Parameters
    ----------
    x_seg    : array
    y_seg    : array
    model_fn : callable   f(x, *params) -> y
    p0_fn    : callable   f(x_seg, y_seg) -> list of initial guesses

    Returns
    -------
    dict with popt, y_fit, r2, success, error
    """
    try:
        p0      = p0_fn(x_seg, y_seg)
        popt, _ = curve_fit(model_fn, x_seg, y_seg, p0=p0, maxfev=10000)
        y_fit   = model_fn(x_seg, *popt)
        ss_res  = np.sum((y_seg - y_fit) ** 2)
        ss_tot  = np.sum((y_seg - y_seg.mean()) ** 2)
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"popt": popt, "y_fit": y_fit, "r2": r2,
                "success": True, "error": None}
    except Exception as e:
        return {"popt": None, "y_fit": np.zeros_like(y_seg),
                "r2": 0.0, "success": False, "error": str(e)}


def compute_auc_from_trace(x, y):
    """
    Trapezoidal area under an already-extracted trace, split by sign.

    The shared core every other AUC entry point (compute_auc_window,
    compute_auc_matrix) delegates to, so the sign-split logic lives in
    exactly one place. positive_auc integrates only where y >= 0
    (negative samples clipped to 0); negative_auc integrates only where
    y <= 0 (positive samples clipped to 0) and is a real *signed* area
    (<= 0), not an absolute value — total_auc is their sum, equivalent to
    the plain net signed integral of y.

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