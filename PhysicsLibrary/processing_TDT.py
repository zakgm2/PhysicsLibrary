"""
processing_TDT.py
-----------------
Signal processing pipeline for TDT (Tucker-Davis Technologies) fibre photometry data.

Handles loading, bleaching correction, denoising, and event marker extraction.
Depends on the `tdt` Python SDK for reading tank files.
"""

import os

import numpy as np
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import tdt

from .models import double_exponential_model as double_exponential

REGRESSION_METHODS = ("ransac", "huber", "ols")


def _robust_linear_fit(x, y, method="ransac"):
    """
    Motion correction regresses the isosbestic (415 nm) stream onto the
    signal (465 nm) stream. An ordinary least-squares fit (np.polyfit) lets
    the exact kind of thing this regression is meant to remove (a burst of
    motion artifact, a fiber-cord twist) drag the fitted line toward
    itself, corrupting the motion-free prediction everywhere else in the
    recording — RANSAC and Huber are two different ways of resisting that;
    OLS is offered as a plain, no-robustness baseline.

    method : {"ransac", "huber", "ols"}
        "ransac" (default): repeatedly fits a line to small random
        subsets, keeps whichever subset gets the most other points within
        residual_threshold of it (the inliers), and does one final fit on
        just those inliers. Points that never look like they belong to the
        same line (i.e., artifacts) are excluded from the fit entirely
        rather than merely downweighted — the right choice when artifacts
        are occasional and severe (a fiber-cord twist, a brief motion
        burst) rather than spread continuously through the recording.

        residual_threshold is set explicitly to 3x a robust noise
        estimate. sklearn's own default (the MAD of y around its median)
        is calibrated for roughly-flat data and comes out far too loose
        here, since y's spread is dominated by the real 415-vs-465 trend
        rather than noise; left at the default, RANSAC would accept every
        point as an inlier and silently degrade to an ordinary
        least-squares fit.

        The noise estimate itself comes from the MAD of an initial
        ordinary least-squares fit's residuals (not from
        consecutive-sample differences, tried previously) — diff-based
        noise estimation measures noise at the timescale of one sample,
        which at TDT's sampling rates (~1 kHz, adjacent samples ~1ms
        apart) is dominated by ADC/thermal noise far smaller than the
        residual scale a real 415-vs-465 regression actually produces once
        genuine biological signal (the very thing this correction is
        meant to preserve, not remove) is accounted for. Confirmed on a
        real recording: diff-based noise estimation set a threshold ~350x
        tighter than the fit's actual residual scale, rejecting 98% of
        samples as "outliers". MAD of the OLS fit's own residuals is on
        the right scale by construction, and stays robust to the
        artifacts it's meant to exclude — they're exactly the kind of
        minority-of-points deviation MAD is insensitive to, same reasoning
        as using MAD over standard deviation anywhere else here.

        "huber": every point stays in the fit, but points far from the
        line (beyond `epsilon` times the residual scale, sklearn's
        default epsilon=1.35) get down-weighted rather than excluded —
        gentler than RANSAC's hard in/out cut, better suited to noise
        that's elevated throughout rather than concentrated in a few bad
        stretches.

        "ols": plain np.polyfit — no robustness at all. Included as a
        baseline / for comparison, not recommended for real motion
        correction (see the module-level RANSAC discussion above for why).

    Returns
    -------
    (a, b, inlier_fraction) : float
        Slope, intercept, and the fraction of points kept as inliers for
        the final fit (1.0 = every point used). RANSAC's is a real
        exclusion fraction; OLS always reports 1.0 (never excludes
        anything); Huber's is 1 - the fraction its own `outliers_` mask
        flags as beyond epsilon, so it stays informative despite Huber
        never truly dropping a point from the fit.
    """
    if method not in REGRESSION_METHODS:
        raise ValueError(f"Unknown regression method {method!r}; expected one of {REGRESSION_METHODS}")

    if method == "ols":
        a, b = np.polyfit(x, y, 1)
        return float(a), float(b), 1.0

    if method == "huber":
        model = HuberRegressor()
        model.fit(x.reshape(-1, 1), y)
        a = float(model.coef_[0])
        b = float(model.intercept_)
        inlier_fraction = float(np.mean(~model.outliers_))
        return a, b, inlier_fraction

    ols_coeffs = np.polyfit(x, y, 1)
    ols_resid = y - np.polyval(ols_coeffs, x)
    # 0.6745 maps MAD to a Gaussian standard deviation.
    noise_est = np.median(np.abs(ols_resid - np.median(ols_resid))) / 0.6745

    # Set the threshold to 3x the robust noise estimate (equivalent to a 3-sigma bound)
    model = RANSACRegressor(residual_threshold=3 * noise_est, random_state=0)
    model.fit(x.reshape(-1, 1), y)

    # Coerce to float to unpack single-element arrays safely
    a = float(model.estimator_.coef_[0])
    b = float(model.estimator_.intercept_)
    inlier_fraction = float(np.mean(model.inlier_mask_))

    return a, b, inlier_fraction


def validate_tdt_folder(path):
    """
    Validate whether a directory contains a TDT recording block.

    Parameters
    ----------
    path : str
        Path to the TDT data folder.

    Returns
    -------
    tuple
        (bool, str)
        - True + folder name if valid
        - False + error message if invalid
    """
    if not path:
        return False, "No folder selected."

    has_tbk = any(fname.endswith('.Tbk') for fname in os.listdir(path))

    if has_tbk:
        return True, os.path.basename(path)
    else:
        return False, "Invalid Folder: No TDT block files (.Tbk) found."


def process_tdt_folder(folder_path, regression_method="ransac"):
    """
    Full photometry processing pipeline for a TDT recording.

    Steps:
    1. Load TDT block
    2. Extract 465 nm (signal) and optional 415 nm (reference)
    3. Perform regression-based motion correction
    4. Correct photobleaching trend
    5. Compute ΔF/F
    6. Denoise signal
    7. Extract event markers

    Parameters
    ----------
    folder_path : str
        Path to TDT recording folder.
    regression_method : {"ransac", "huber", "ols"}
        Which linear regression the isosbestic-onto-signal motion
        correction (step 3) uses — see _robust_linear_fit's own docstring
        for what each one actually does. Defaults to "ransac", the most
        robust of the three against occasional severe artifacts.

    Returns
    -------
    dict
        Processed signals and metadata:
        - x: time vector
        - raw: corrected fluorescence signal
        - corr: final ΔF/F (denoised)
        - dff: same as corr (alias)
        - f0: bleaching baseline
        - fs: sampling frequency
        - store: signal label
        - markers: behavioral event markers
        - channels: list of {"key", "label", "y"} for each raw
          per-wavelength stream found (always "main_driver"; "isosbestic"
          too if a 415 reference stream exists) — the un-motion-corrected,
          un-normalized traces, for a caller that wants to plot/analyze
          one of them directly instead of only "raw"/"corr" above
        - motion_correction_inlier_fraction: fraction (0-1) of samples
          kept as inliers during motion correction (see
          _robust_linear_fit for what this means per regression_method),
          or None if there was no 415 reference stream to correct against
    """
    data_struct = get_tdt_struct(folder_path)
    streams = data_struct.streams.keys()

    name_465 = next((s for s in streams if '465' in s), None)
    name_415 = next((s for s in streams if '415' in s), None)

    if not name_465:
        raise ValueError("No 465 signal found")

    x, y_465, fs = get_plot_data(data_struct, name_465)

    if name_415:
        _, y_415, _ = get_plot_data(data_struct, name_415)
        a, b, inlier_fraction = _robust_linear_fit(y_415, y_465, method=regression_method)
        y_fit = a * y_415 + b
        y_final = y_465 - y_fit
        display_name = f"Corrected {name_465} (via {name_415})"
    else:
        y_final = y_465
        display_name = f"{name_465} (Uncorrected)"
        inlier_fraction = None

    _, trend = correct_bleaching(y_final, fs)

    f0  = np.maximum(trend, 1e-6)
    dff = (y_final - f0) / f0
    dff = denoise_signal(dff, fs, cutoff=5)

    # Raw per-wavelength channels, for callers that want to plot/analyze
    # the main driver (probe) or isosbestic (control) trace on its own
    # instead of only the motion-corrected/normalized result above.
    # Nothing about stream naming or count is hardcoded beyond the 465/415
    # detection already done above — a block with no 415 reference simply
    # has no 'isosbestic' entry, so this works for any experimental setup
    # this function already supports.
    channels = [
        {"key": "main_driver", "label": f"Main Driver ({name_465})", "y": y_465},
    ]
    if name_415:
        channels.append({"key": "isosbestic", "label": f"Isosbestic ({name_415})", "y": y_415})

    return {
        "x":        x,
        "raw":      y_final,
        "corr":     dff,
        "dff":      dff,
        "f0":       f0,
        "fs":       fs,
        "store":    display_name,
        "markers":  get_event_markers(data_struct),
        "channels": channels,
        "motion_correction_inlier_fraction": inlier_fraction,
    }


def get_tdt_struct(path):
    """
    Load a Tucker-Davis Technologies (TDT) recording block.

    Streams and scalars are read in a single call (unaffected by the issue
    below). Epoc stores are auto-discovered from the block's own headers
    and then read ONE STORE AT A TIME, each with its own fresh
    tdt.read_block() call:

    - The `tdt` SDK has a bug where reading multiple epoc stores together
      breaks if any store has a mismatched onset/offset count (e.g. a
      strobe/epoch that was still active when the recording was stopped —
      a completely normal way for a session to end). Reading one store per
      call sidesteps it; a store that still fails to a genuine data issue
      is skipped with a warning rather than crashing the entire load.
    - Store names must come from each store's own `.name` attribute, not
      the header dict's key — TDT sanitizes store codes containing `/`
      into a trailing-underscore Python attribute name (e.g. dict key
      "EE1_" but the real store code passed to `store=` is "EE1/").
    - Epoc reads must NOT reuse the cached `headers=` object from the
      initial header scan — doing so carries over the same mismatched
      onset/offset state that causes the crash in the first place. Only
      the streams/scalars read benefits from header reuse.

    This entirely auto-discovers whatever stores a given recording
    contains — nothing about store names/types is hardcoded, so it works
    for any TDT block regardless of how it was configured in Synapse.

    Parameters
    ----------
    path : str
        Folder containing TDT data.

    Returns
    -------
    object
        Parsed TDT data structure (data.streams, data.epocs, data.scalars).
    """
    import warnings

    heads = tdt.read_block(path, headers=1)
    if heads is None:
        raise Exception("TDT returned an empty object.")

    data = tdt.read_block(path, headers=heads, evtype=['streams', 'scalars'], verbose=0)
    if data is None:
        raise Exception("TDT returned an empty object.")

    epoc_stores = [(key, s.name) for key, s in heads.stores.items()
                   if getattr(s, 'type_str', None) == 'epocs']

    for key, real_name in epoc_stores:
        try:
            # No headers= reuse here — see docstring.
            ep = tdt.read_block(path, store=real_name, evtype=['epocs'], verbose=0)
        except Exception as e:
            warnings.warn(
                f"Skipping epoc store '{key}' ({real_name}): {e}",
                RuntimeWarning, stacklevel=2,
            )
            continue
        if not ep.epocs:
            continue
        store_key = next(iter(ep.epocs.keys()))
        setattr(data.epocs, key, ep.epocs[store_key])

    return data


def get_plot_data(data, store_name, channel=0, max_points=None):
    """
    Extract time-series data from a TDT stream.

    Parameters
    ----------
    data : object
        TDT data structure
    store_name : str
        Stream name (e.g., 'x465A')
    channel : int
        Channel index
    max_points : int or None
        Optional downsampling limit

    Returns
    -------
    tuple
        (time array, signal array, sampling frequency)
    """
    stream = data.streams[store_name]
    fs     = stream.fs

    data_2d = np.atleast_2d(stream.data)
    if channel >= data_2d.shape[0]:
        channel = 0

    y_full = data_2d[channel, :]

    if max_points:
        ds_factor = max(1, len(y_full) // max_points)
        y = y_full[::ds_factor]
        x = np.arange(len(y)) * (ds_factor / fs)
    else:
        y = y_full
        x = np.arange(len(y)) / fs

    return x, y, fs


def correct_bleaching(y, fs):
    """
    Estimate and correct photobleaching using masked curve fitting.

    Returns
    -------
    corrected : array
        Bleaching-corrected signal
    trend : array
        Estimated baseline trend
    """
    x = np.arange(len(y)) / fs

    threshold = np.median(y)
    mask = y > threshold
    x_fit, y_fit = x[mask], y[mask]

    if len(y_fit) < 100:
        return y, np.zeros_like(y)

    k_guess   = np.percentile(y_fit, 10)
    total_amp = np.max(y_fit) - k_guess
    p0        = [total_amp * 0.6, 0.05, total_amp * 0.4, 0.0001, k_guess]
    lower     = [0, 0, 0, 0, k_guess * 0.8]
    upper     = [np.inf, 1, np.inf, 0.1, np.max(y_fit)]

    try:
        popt, _ = curve_fit(double_exponential, x_fit, y_fit, p0=p0,
                            bounds=(lower, upper), maxfev=10000)
        trend = double_exponential(x, *popt)
    except Exception:
        # curve_fit failed — fall back to log-linear fit as a rough trend estimate
        import warnings
        warnings.warn(
            "correct_bleaching: double-exponential fit failed; falling back to log-linear trend.",
            RuntimeWarning, stacklevel=2,
        )
        coeffs = np.polyfit(x_fit, np.log(np.maximum(y_fit, 1e-6)), 1)
        trend  = np.exp(np.polyval(coeffs, x))

    corrected = y - trend + trend[0]
    return corrected, trend


def get_event_markers(data):
    """
    Extract behavioral event markers from every populated TDT epoc store.

    A recording can easily have a dozen+ epoc stores (I/O strobes, Epoch
    Event Storage, free-text notes, ...) — every one of them becomes its
    own marker group here, tagged by store name via the 'store' key, so
    a caller (e.g. a GUI) can let the user toggle each store's markers
    independently instead of dumping every store onto the plot at once
    (which overlaps into an unreadable mess). The one exception is the
    'Tick' store — TDT's own 1-second heartbeat/reference signal, not a
    behavioral event — which is skipped entirely.

    Every epoc is a state that goes high (onset) and later low (offset) —
    for a lever press that's press/release, for a pump or light that's
    on/off. Both are emitted here, each tagged with a 'phase' key ('high'
    for onset, 'low' for offset), so a caller can decide per store whether
    it cares about both edges (e.g. how long the pump ran) or only one
    (e.g. just the press, not the release) instead of that decision being
    baked into the library.

    The 'Note' store keeps its original behaviour: each onset becomes a
    marker labeled with the actual note text (from Notes.txt), and has no
    offset/'low' marker — free-text annotations are instantaneous, not a
    state with a duration. Every other epoc store gets one 'high' marker
    per onset and one 'low' marker per offset, both labeled with the store
    name.

    Nothing about which stores exist is hardcoded — this walks whatever
    data.epocs actually contains, so it works for any TDT block.

    Returns
    -------
    list of dict
        Each dict contains:
        - time  : float
        - label : str
        - color : str
        - store : str   (source epoc store name, for grouping/toggling)
        - phase : str   ('high'=onset, 'low'=offset; omitted for Note markers)
    """
    if not hasattr(data, 'epocs'):
        return []

    # Experiment-specific label→colour mapping for Note text; every other
    # store cycles through a fixed palette so each store gets a distinct,
    # consistent colour across redraws.
    note_color_map = {'Clap': 'red', 'Sucrose': 'green', 'Stop': 'blue'}
    palette = ['black', 'purple', 'orange', 'saddlebrown', 'teal', 'magenta',
               'darkgreen', 'navy', 'gray', 'olive']

    markers = []
    store_keys = sorted(k for k in data.epocs.keys())

    for i, store_key in enumerate(store_keys):
        display_name = store_key.rstrip('_')  # cosmetic: TDT pads slash-containing codes with a trailing "_"
        if display_name.lower() == 'tick':
            # TDT's own 1-second heartbeat/reference signal, not a
            # behavioral event — every recording has it, it's never
            # something a caller wants to plot, analyze, or splice
            # against, and its sheer regularity would otherwise flood
            # every marker/event picker in the GUI.
            continue

        epoc = data.epocs[store_key]
        onsets = getattr(epoc, 'onset', [])
        if len(onsets) == 0:
            continue

        if store_key == 'Note':
            notes = getattr(epoc, 'notes', [])
            for n, t in zip(notes, onsets):
                note_str = n.decode() if isinstance(n, bytes) else str(n)
                note_str = note_str.strip()
                markers.append({
                    'time':  float(t),
                    'label': note_str,
                    'color': note_color_map.get(note_str, 'black'),
                    'store': display_name,
                })
        else:
            color = palette[i % len(palette)]
            for t in onsets:
                # A level-type epoc store (buffered, tracks a logic
                # signal's on/off state — common for some Synapse Gizmo
                # outputs) has no history before recording starts; if its
                # logic already happened to be high the instant recording
                # began, TDT inserts a synthetic onset at exactly t=0 to
                # represent that pre-existing state, not a real event.
                # Mirrors the offset=inf guard below for the same store
                # type's opposite edge case (still active at the end).
                if t == 0.0:
                    continue
                markers.append({
                    'time':  float(t),
                    'label': display_name,
                    'color': color,
                    'store': display_name,
                    'phase': 'high',
                })
            offsets = getattr(epoc, 'offset', [])
            for t in offsets:
                # TDT represents a strobe still active at recording end with
                # offset=inf — not a real timestamp, nothing to plot.
                if not np.isfinite(t):
                    continue
                markers.append({
                    'time':  float(t),
                    'label': display_name,
                    'color': color,
                    'store': display_name,
                    'phase': 'low',
                })

    return markers


def debounce_events(times, min_isi):
    """
    Collapse switch-bounce / double-tap duplicates out of a list of event
    timestamps (e.g. lever-press onsets), independent of any fixed-ratio
    schedule (FR1, FR3, ...).

    A real lever contact and a bounce/duplicate reading of the same
    physical press land within milliseconds of each other — far closer
    than an animal can genuinely press again. Sorting the timestamps and
    dropping anything within `min_isi` seconds of the last *kept* event
    removes those duplicates while leaving genuinely fast consecutive
    presses (e.g. during an FR3 burst) intact, as long as `min_isi` is
    below the animal's real max press rate.

    Parameters
    ----------
    times : sequence of float
        Raw event timestamps, in seconds. Need not be sorted.
    min_isi : float
        Minimum inter-event interval, in seconds. Any event whose gap from
        the previous kept event is smaller than this is dropped. The
        first press of a bounce cluster is always the one kept.

    Returns
    -------
    list of float
        Sorted timestamps with sub-`min_isi` duplicates removed.
    """
    sorted_times = sorted(float(t) for t in times)
    kept = []
    for t in sorted_times:
        if not kept or (t - kept[-1]) >= min_isi:
            kept.append(t)
    return kept


def denoise_signal(signal, fs, cutoff=5, order=2):
    """
    Low-pass Butterworth filter for ΔF/F signals.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz
    order : int
        Filter order

    Returns
    -------
    array
        Filtered signal
    """
    nyquist       = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a          = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)