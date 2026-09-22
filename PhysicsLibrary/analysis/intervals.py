"""
analysis/intervals.py
-----------------------
Time-since-previous-marker, for the Measure Intervals tool.
"""


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
