"""
splice.py
---------
Non-destructive time-range edits of a signal: trim to a range, or cut
one out and stitch the remainder back together. Pure array/dict
operations — no knowledge of any GUI's cache/context shape, just plain
x/raw/corr arrays and marker dicts (each with at least a 'time' key).

Two operations:
  - keep_inside: the usual "trim to a window" splice.
  - cut_out: removes a range from the middle and stitches the two
    remaining pieces together. Everything after the cut is shifted
    backward by the cut's duration so the timeline stays contiguous —
    a gap in x would break every downstream analysis that assumes
    uniform sampling (PETH windows, FFT, ...) right at the cut boundary.
    Markers inside the removed range are dropped (that moment no longer
    exists); markers after it are shifted by the same amount so they
    stay aligned with the signal.
"""

import numpy as np


def slice_markers_keep_inside(markers, start, end):
    """Keeps original absolute timestamps (not re-zeroed to the splice
    start) so every downstream analysis that reads times straight from
    the marker dicts keeps working with no special-casing."""
    return [dict(m) for m in markers if start <= m['time'] <= end]


def slice_markers_cut_out(markers, start, end, shift):
    """Drops markers inside the removed range; shifts the rest that fall
    after it by `shift` (the cut's duration) to stay aligned with the
    signal's new, contiguous timeline."""
    out = []
    for m in markers:
        t = m['time']
        if start < t < end:
            continue
        m = dict(m)
        if t >= end:
            m['time'] = t - shift
        out.append(m)
    return out


def splice_keep_inside(x, raw, corr, markers, detected_markers, start, end, extra_channels=None):
    """
    Trims x/raw/corr to [start, end] and filters both marker lists to
    the same range.

    Parameters
    ----------
    x, raw, corr : array
    markers, detected_markers : list of dict, each with a 'time' key
    start, end : float
    extra_channels : dict of {name: array}, optional
        Any other arrays aligned sample-for-sample with x on their LAST
        axis (e.g. a recording's raw per-wavelength channels — main
        driver, isosbestic — each 1D; or a multi-detector array like
        Oxysoft's o2hb/hhb, shape (n_detectors, n_samples)) that also
        need trimming to the same range. Sliced along axis=-1 so both
        1D and 2D arrays work with no special-casing. Not required — a
        caller with only x/raw/corr can omit it entirely.

    Returns
    -------
    dict with x, raw, corr (copies, not views), markers,
    detected_markers, n_samples, extra_channels (trimmed the same way,
    {} if none were given) — or None if the range doesn't contain at
    least 2 samples.
    """
    i0 = int(np.searchsorted(x, start, side='left'))
    i1 = int(np.searchsorted(x, end, side='right'))
    if i1 - i0 < 2:
        return None

    return {
        "x": x[i0:i1].copy(),
        "raw": raw[i0:i1].copy(),
        "corr": corr[i0:i1].copy(),
        "markers": slice_markers_keep_inside(markers, start, end),
        "detected_markers": slice_markers_keep_inside(detected_markers, start, end),
        "n_samples": i1 - i0,
        "extra_channels": {name: y[..., i0:i1].copy() for name, y in (extra_channels or {}).items()},
    }


def splice_cut_out(x, raw, corr, markers, detected_markers, start, end, extra_channels=None):
    """
    Removes [start, end] from x/raw/corr and stitches the remainder
    together, shifting everything after the cut backward by the cut's
    duration so the timeline stays contiguous. Markers inside the cut
    are dropped; markers after it are shifted by the same amount.

    Parameters
    ----------
    x, raw, corr : array
    markers, detected_markers : list of dict, each with a 'time' key
    start, end : float
    extra_channels : dict of {name: array}, optional
        Any other arrays aligned sample-for-sample with x on their LAST
        axis that also need the same range cut out and stitched — see
        splice_keep_inside (1D and 2D, e.g. Oxysoft's multi-detector
        arrays, both work).

    Returns
    -------
    dict with x, raw, corr, markers, detected_markers, n_samples,
    extra_channels (cut/stitched the same way, {} if none were given) —
    or None if there isn't usable signal on both sides of the cut to
    stitch together.
    """
    i0 = int(np.searchsorted(x, start, side='left'))
    i1 = int(np.searchsorted(x, end, side='right'))
    if i0 < 1 or i1 >= len(x) - 1 or i1 <= i0:
        return None

    shift = float(x[i1] - x[i0])  # cut duration, used to reconnect the timeline
    new_x = np.concatenate([x[:i0], x[i1:] - shift])

    return {
        "x": new_x,
        "raw": np.concatenate([raw[:i0], raw[i1:]]),
        "corr": np.concatenate([corr[:i0], corr[i1:]]),
        "markers": slice_markers_cut_out(markers, start, end, shift),
        "detected_markers": slice_markers_cut_out(detected_markers, start, end, shift),
        "n_samples": len(new_x),
        "extra_channels": {name: np.concatenate([y[..., :i0], y[..., i1:]], axis=-1)
                            for name, y in (extra_channels or {}).items()},
    }
