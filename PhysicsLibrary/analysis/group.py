"""
analysis/group.py
-----------------
Group analysis across recordings ("Hypothesis Testing" in PyAT): the same
event-locked responses measured in every recording of a group, reduced to a few
numbers per trial, and compared with a linear mixed-effects model.

What is analysed
    Every recording in the group is one subject. The variables are event markers
    (lever presses, notes, ...); a trial is one occurrence of a marker, taken from
    the recording exactly as the single-recording Event PETH takes it. Each trial
    is a window around its event: [-pre, +post] seconds. Inside it, a baseline
    window and a response window (both in seconds relative to the event) define
    the numbers computed per trial:

        auc      area under the baseline-corrected trace across the response window
        peak     baseline-corrected value at the peak in the response window
                 (peak_direction: "positive", "negative", or "absolute" = the largest
                 deflection either way, signed, as the single-recording Event PETH does)
        mean     mean of the baseline-corrected trace across the response window
        latency  seconds from the event to the peak
        decay    seconds from the peak until the trace has fallen to
                 `decay_fraction` of the peak (0.5 = half-decay time); looked for from
                 the peak to the end of the trial window, empty if it never falls that far

    "Baseline-corrected" is set by `signal`: "dff" subtracts the baseline window's
    mean from the trial's dF/F; "zscore" also divides by the baseline's SD. The signal is
    first smoothed with a moving average of `smooth_seconds`, as Event PETH does.

How it is analysed (group_stats.py)
    Each measure is analysed on its own. With two or more markers: a linear mixed-effects
    model, measure ~ marker, with a random intercept for subject and a random intercept for
    each subject x marker cell. The cell term matters: trials of one subject and marker
    share a cell-level effect (and neighbouring trials share signal), and without it the
    test treats every trial as independent evidence about the marker difference, which gives
    false positives far above the nominal rate (65-93% of runs at a nominal 5% in the simulations
    described in group_stats.py).
    p-values use t / F with subject-based degrees of freedom (n_subjects - 1 for a contrast,
    like a paired test), then all pairwise comparisons between markers are corrected for
    multiple comparisons. With one marker there is nothing to compare it with: AUC and mean
    amplitude are tested against zero (the baseline-corrected trace has mean zero without a
    response); the peak, latency and decay time, which are extreme-value or timing statistics
    with no zero to compare against, are reported as estimates with intervals only. Every
    analysis is repeated on subject x marker means as a check that does not rely on trials
    being independent.

This file has the parts needed before any signal is loaded: the settings object
(GroupSpec), and checks on the design (which markers the recordings share, how many
trials each subject would contribute, whether trials overlap). group_trials.py slices the
trials and computes the measures; group_stats.py fits the models.

Defaults
    There is no formal standard for the window. The closest thing to a reference is
    TDT's own fiber photometry epoch-averaging example, which uses TRANGE = [-10, 20]
    (start, then window *duration*: -10 s to +10 s around the event) and a baseline of
    -10 to -6 s, z-scoring each trial against it. GuPPy and pMAT leave the window and
    baseline to the user. These defaults follow TDT's example; the response window is
    the whole post-event half.
"""

import math
from dataclasses import dataclass, field, fields

import numpy as np

from ..processing_TDT import REGRESSION_METHODS

METRICS = ("auc", "peak", "mean", "latency", "decay")
METRIC_LABELS = {
    "auc":     "AUC",
    "peak":    "Peak amplitude",
    "mean":    "Mean amplitude",
    "latency": "Latency to peak",
    "decay":   "Decay time",
}
SIGNALS = ("dff", "zscore")
PEAK_DIRECTIONS = ("positive", "negative", "absolute")
CORRECTIONS = ("holm", "bonferroni", "fdr_bh")
CORRECTION_LABELS = {"holm": "Holm", "bonferroni": "Bonferroni", "fdr_bh": "Benjamini-Hochberg (FDR)"}

DEFAULT_PRE = 10.0
DEFAULT_POST = 10.0
DEFAULT_BASELINE = (-10.0, -6.0)
DEFAULT_RESPONSE = (0.0, 10.0)

# Below this many subjects a mixed model's estimate of how much subjects differ is unreliable;
# below the second, its Wald z p-values (no small-sample degrees of freedom) run optimistic.
MIN_SUBJECTS_RELIABLE = 6
MIN_SUBJECTS_APPROXIMATE_P = 20


@dataclass
class GroupSpec:
    """Everything the user decides for one group analysis; saved as JSON next to the results."""

    group_name: str = ""
    # [{"subject": name, "folder": path}, ...] — one entry per recording, subject names unique.
    subjects: list = field(default_factory=list)
    # Marker names as PyAT lists them: a store's name plus a superscript 1 (onset) or 0 (offset),
    # or a note's own text.
    markers: list = field(default_factory=list)
    # Store renames in effect when the markers were listed ({store id: shown name}); a marker name
    # only means something together with the renames that produced it.
    store_labels: dict = field(default_factory=dict)
    pre: float = DEFAULT_PRE
    post: float = DEFAULT_POST
    baseline: tuple = DEFAULT_BASELINE
    response: tuple = DEFAULT_RESPONSE
    signal: str = "dff"
    # Moving-average smoothing of the dF/F before slicing, in seconds (0 = none); 0.5 is what the
    # single-recording Event PETH applies.
    smooth_seconds: float = 0.5
    metrics: list = field(default_factory=lambda: list(METRICS))
    peak_direction: str = "absolute"
    decay_fraction: float = 0.5
    # Multiple-comparison correction across the pairwise comparisons of one measure, and the level.
    correction: str = "holm"
    alpha: float = 0.05
    # The dF/F is recomputed for every recording with this motion-correction method.
    regression_method: str = "ols"

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["baseline"] = list(self.baseline)
        d["response"] = list(self.response)
        return d

    @classmethod
    def from_dict(cls, d):
        """Rebuilds a spec from to_dict()'s output (a JSON round trip); unknown keys are ignored."""
        known = {f.name for f in fields(cls)}
        spec = cls(**{k: v for k, v in d.items() if k in known})
        spec.baseline = tuple(spec.baseline)
        spec.response = tuple(spec.response)
        return spec

    def window_problems(self):
        """Just the rules for the window around the event, its baseline and its response window."""
        out = []
        if not (_finite(self.pre) and self.pre > 0 and _finite(self.post) and self.post > 0):
            out.append("The window needs a positive number of seconds before and after the event.")
            return out
        b0, b1 = self.baseline
        if not (_finite(b0) and _finite(b1) and b0 < b1):
            out.append("The baseline window must run from an earlier to a later time.")
        elif b0 < -self.pre or b1 > 0:
            out.append(f"The baseline window must lie between -{self.pre:g} s and 0 s "
                       "(before the event, inside the window).")
        r0, r1 = self.response
        if not (_finite(r0) and _finite(r1) and r0 < r1):
            out.append("The response window must run from an earlier to a later time.")
        elif r0 < 0 or r1 > self.post:
            out.append(f"The response window must lie between 0 s and {self.post:g} s "
                       "(after the event, inside the window).")
        return out

    def problems(self):
        """Everything that makes this spec unusable, as plain sentences (empty list = fine)."""
        out = []
        names = [s.get("subject", "") for s in self.subjects]
        if len(self.subjects) < 2:
            out.append("Pick at least two recordings: a group comparison needs more than one subject.")
        if any(not str(n).strip() for n in names):
            out.append("Every subject needs a name.")
        elif len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            out.append("Subject names must be unique (repeated: " + ", ".join(dupes) + ").")

        if not self.markers:
            out.append("Pick at least one marker.")
        elif len(set(self.markers)) != len(self.markers):
            out.append("Each marker can only be picked once.")

        out.extend(self.window_problems())

        if not self.metrics:
            out.append("Pick at least one measure (AUC, peak, ...).")
        elif any(m not in METRICS for m in self.metrics):
            out.append("Unknown measure: " + ", ".join(m for m in self.metrics if m not in METRICS) + ".")
        if self.signal not in SIGNALS:
            out.append(f"Signal must be one of {', '.join(SIGNALS)}.")
        if self.peak_direction not in PEAK_DIRECTIONS:
            out.append(f"Peak direction must be one of {', '.join(PEAK_DIRECTIONS)}.")
        if not (_finite(self.decay_fraction) and 0 < self.decay_fraction < 1):
            out.append("The decay level must be between 0 and 100% of the peak.")
        if not (_finite(self.smooth_seconds) and self.smooth_seconds >= 0):
            out.append("The smoothing window must be zero (none) or a positive number of seconds.")
        if self.correction not in CORRECTIONS:
            out.append(f"Correction must be one of {', '.join(CORRECTIONS)}.")
        if not (_finite(self.alpha) and 0 < self.alpha < 1):
            out.append("The significance level must be between 0 and 1.")
        if self.regression_method not in REGRESSION_METHODS:
            out.append(f"Regression method must be one of {', '.join(REGRESSION_METHODS)}.")
        return out


def _finite(v):
    return isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(v)


def marker_index(scans):
    """
    Which markers the recordings have, and how many events of each.

    Parameters
    ----------
    scans : list of {"subject": str, "groups": {marker name: [event times]}, ...}

    Returns
    -------
    {marker: {"recordings": how many recordings have it, "events": total events,
              "per_subject": {subject: events}}} — only recordings that have at least one
    event of a marker are counted for it.
    """
    index = {}
    for sc in scans:
        for name, times in sc["groups"].items():
            if len(times) == 0:
                continue
            entry = index.setdefault(name, {"recordings": 0, "events": 0, "per_subject": {}})
            entry["recordings"] += 1
            entry["events"] += len(times)
            entry["per_subject"][sc["subject"]] = len(times)
    return index


def common_markers(scans):
    """Marker names present (at least one event) in every one of the recordings, sorted."""
    n = len(scans)
    return sorted(name for name, e in marker_index(scans).items() if n and e["recordings"] == n)


def design_summary(scans, markers, pre, post):
    """
    How many trials each subject would contribute per marker, worked out from the event
    times alone (no signal is loaded), plus the problems worth knowing about first.

    A trial is *usable* if its whole window, [event - pre, event + post], lies inside the
    recording. Trials *overlap* when their windows share samples (events less than
    pre + post seconds apart): the same stretch of signal then enters more than one trial,
    so trials of one subject are not independent of each other.

    Parameters
    ----------
    scans : list of {"subject", "groups": {marker: [event times]}, "t_range": (first, last)
            sample time, or None if unknown}
    markers : list of marker names to summarise
    pre, post : float
        Seconds before and after each event.

    Returns
    -------
    dict with
        cells : {subject: {marker: {"events", "usable", "overlapping"}}} ("overlapping"
            counts usable trials whose window shares samples with another usable trial's)
        markers : {marker: {"events", "usable", "overlapping", "subjects_with_trials",
            "min_usable", "max_usable"}} (min/max across subjects, zeros included)
        n_subjects : number of subjects
        warnings : list of {"level": "warning" | "info", "text": str}
    """
    span = pre + post
    cells = {}
    for sc in scans:
        t0, t1 = sc.get("t_range") or (-math.inf, math.inf)
        row = {}
        for m in markers:
            times = np.sort(np.asarray(sc["groups"].get(m, []), dtype=float))
            usable = times[(times - pre >= t0) & (times + post <= t1)]
            overlap = np.zeros(len(usable), dtype=bool)
            if len(usable) > 1:
                close = np.diff(usable) < span
                overlap[:-1] |= close
                overlap[1:] |= close
            row[m] = {"events": int(len(times)), "usable": int(len(usable)), "overlapping": int(overlap.sum())}
        cells[sc["subject"]] = row

    per_marker = {}
    for m in markers:
        usable = [cells[s][m]["usable"] for s in cells]
        per_marker[m] = {
            "events": sum(cells[s][m]["events"] for s in cells),
            "usable": sum(usable),
            "overlapping": sum(cells[s][m]["overlapping"] for s in cells),
            "subjects_with_trials": sum(1 for u in usable if u > 0),
            "min_usable": min(usable) if usable else 0,
            "max_usable": max(usable) if usable else 0,
        }

    warnings = []
    n_subjects = len(scans)
    for m in markers:
        info = per_marker[m]
        missing = [s for s in cells if cells[s][m]["usable"] == 0]
        if info["usable"] == 0:
            warnings.append({"level": "warning", "text": f"'{m}' has no usable trials in any recording."})
        elif missing:
            shown = ", ".join(missing[:4]) + (f" and {len(missing) - 4} more" if len(missing) > 4 else "")
            warnings.append({"level": "warning",
                             "text": f"'{m}': {len(missing)} of {n_subjects} subjects have no usable trials "
                                     f"({shown}); they will not contribute to it."})
        if info["usable"]:
            share = info["overlapping"] / info["usable"]
            if share >= 0.5:
                warnings.append({"level": "warning",
                                 "text": f"'{m}': {share:.0%} of trials share samples with another trial's window, "
                                         "so trials are not independent of each other. A shorter window reduces this."})
            elif share > 0:
                warnings.append({"level": "info",
                                 "text": f"'{m}': {share:.0%} of trials share samples with another trial's window."})
    if n_subjects < MIN_SUBJECTS_RELIABLE:
        warnings.append({"level": "warning",
                         "text": f"Only {n_subjects} subjects: a mixed model cannot estimate how much subjects "
                                 "differ reliably with so few."})
    elif n_subjects < MIN_SUBJECTS_APPROXIMATE_P:
        warnings.append({"level": "info",
                         "text": f"With {n_subjects} subjects the mixed model's p-values (Wald z tests, no small-sample "
                                 "correction) are approximate and lean optimistic."})
    return {"cells": cells, "markers": per_marker, "n_subjects": n_subjects, "warnings": warnings}
