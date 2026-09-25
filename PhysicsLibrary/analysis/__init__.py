"""
analysis
--------
Format-agnostic analysis routines for PyAT, one file per
tool (mirrors physicsanalysis_qt/analysis/'s own layout):

  shared.py       - estimate_sample_rate, mean_channels, smooth_signal
                    (used by more than one tool below)
  zscore_peth.py  - single-click Z-Score PETH (get_zscore_slice, bin_for_heatmap)
  event_peth.py   - stacked-heatmap + trial-average PETH: compute_event_zscore_peth,
                    compute_group_stats, compute_auc_matrix, compute_peri_event_*
  auc.py          - area under curve (compute_auc_from_trace, compute_auc_window)
  fft.py          - compute_fft_slice, find_fft_peaks (peak *drawing* is
                    physicsanalysis_qt's own concern, not this library's —
                    see that repo's analysis/fft.py)
  peak_finder.py  - find_significant_peaks, find_peak_near_events
  curve_fit.py    - compute_slope_segment, fit_model_to_segment
  intervals.py    - compute_marker_intervals

This __init__ just re-exports every public name under the same
`PhysicsLibrary.analysis` namespace the single-file version used to
provide, so PhysicsLibrary/__init__.py's own `from .analysis import ...`
line — and every external caller — didn't need to change.
"""

from .shared import (
    estimate_sample_rate,
    mean_channels,
    smooth_signal,
)
from .zscore_peth import (
    get_zscore_slice,
    bin_for_heatmap,
)
from .event_peth import (
    compute_group_stats,
    compute_event_zscore_peth,
    compute_auc_matrix,
    compute_peri_event_from_trace,
    compute_peri_event_matrix,
)
from .auc import (
    compute_auc_from_trace,
    compute_auc_window,
)
from .fft import (
    compute_fft_slice,
    find_fft_peaks,
)
from .peak_finder import (
    find_significant_peaks,
    find_peak_near_events,
)
from .curve_fit import (
    compute_slope_segment,
    fit_model_to_segment,
)
from .intervals import (
    compute_marker_intervals,
)
