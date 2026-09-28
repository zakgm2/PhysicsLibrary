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
  group.py        - group analysis across recordings: GroupSpec (the settings), marker_index /
                    common_markers / design_summary (what the recordings share, trials per subject)
  group_trials.py - extract_group_trials: the window around each event of a marker, baseline-corrected,
                    reduced to AUC / peak / mean / latency / decay time
  group_stats.py  - fit_group_models: mixed-effects models and pairwise comparisons (GroupResults)
  group_report.py - write_group_results, report_text: the tables as CSV and a text report

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
from .group import (
    GroupSpec,
    marker_index,
    common_markers,
    design_summary,
)
from .group_trials import (
    extract_group_trials,
    measures_for_trial,
    decay_time,
)
from .group_stats import (
    fit_group_models,
    GroupResults,
)
from .group_report import (
    write_group_results,
    report_text as group_report_text,
)
