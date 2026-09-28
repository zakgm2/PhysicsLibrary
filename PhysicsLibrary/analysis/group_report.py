"""
analysis/group_report.py
------------------------
A group analysis written down: every table as a CSV file, and a plain-text report (design, results per
measure, notes, and a methods paragraph to adapt). Files are UTF-8 with a byte-order mark so Excel reads the
superscript 1 / 0 in marker names correctly.
"""

import math
import os

import numpy as np
import pandas as pd

from .group import CORRECTION_LABELS, METRIC_LABELS

REPORT_FILENAME = "analysis_report.txt"

_SIGNAL_TEXT = {"dff": "dF/F minus the baseline mean", "zscore": "z-score against the baseline"}
_DIRECTION_TEXT = {"positive": "the largest positive value", "negative": "the largest negative value",
                   "absolute": "the largest deflection in either direction (signed)"}

CSV_NAMES = {
    "trials": "trials_long.csv",
    "subject_means": "subject_means.csv",
    "descriptives": "descriptives.csv",
    "models": "models.csv",
    "fixed_effects": "model_fixed_effects.csv",
    "omnibus": "model_omnibus_tests.csv",
    "variance": "model_variance_components.csv",
    "estimated_means": "estimated_means.csv",
    "pairwise": "posthoc_pairwise_comparisons.csv",
    "diagnostics": "model_residuals.csv",
    "random_effects": "model_random_effects.csv",
}


def measure_units(measure, signal):
    """Units of a measure, as text: the trace's own unit for amplitudes, its unit x seconds for the AUC."""
    trace = "dF/F" if signal == "dff" else "z"
    return {"auc": f"{trace} x s", "peak": trace, "mean": trace, "latency": "s", "decay": "s"}[measure]


def trace_frame(results):
    """The mean traces in long format: marker, subject, n_trials, time, value."""
    grid = results.trace_grid
    rows = []
    for subject, by_marker in results.traces.items():
        for marker, tr in by_marker.items():
            if tr["n"] == 0:
                continue
            rows.append(pd.DataFrame({"marker": marker, "subject": subject, "n_trials": tr["n"],
                                      "time": grid, "value": tr["mean"]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["marker", "subject", "n_trials", "time", "value"])


def write_group_results(results, directory):
    """Write every table as CSV and the text report into `directory` (created if needed). Returns the paths."""
    os.makedirs(directory, exist_ok=True)
    paths = []
    frames = results.frames()
    for name, filename in CSV_NAMES.items():
        df = frames.get(name)
        if df is None or not len(df):
            continue
        path = os.path.join(directory, filename)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    traces = trace_frame(results)
    if len(traces):
        path = os.path.join(directory, "mean_traces.csv")
        traces.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    path = os.path.join(directory, REPORT_FILENAME)
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(report_text(results))
    paths.append(path)
    return paths


def _num(v, digits=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{v:.{digits}g}"


def _p(v):
    if v is None or not np.isfinite(v):
        return "-"
    return "<0.0001" if v < 1e-4 else f"{v:.4f}"


def _peq(v):
    """'p = 0.0123' or 'p < 0.0001'."""
    text = _p(v)
    return f"p {text}" if text.startswith("<") else f"p = {text}"


def _table(df, columns, headers, indent=6):
    if not len(df):
        return " " * indent + "(none)\n"
    shown = pd.DataFrame({h: df[c] for c, h in zip(columns, headers)})
    return "\n".join(" " * indent + line for line in shown.to_string(index=False, justify="left").splitlines()) + "\n"


def report_text(results):
    """The plain-text report."""
    spec = results.spec
    out = []
    add = out.append
    subjects = [s["subject"] for s in spec.subjects]
    add(f"GROUP ANALYSIS: {spec.group_name}")
    add("=" * (16 + len(spec.group_name)))
    add(f"Subjects ({len(subjects)}): " + ", ".join(subjects))
    add(f"Markers ({len(spec.markers)}): " + ", ".join(spec.markers))
    add(f"Window: {spec.pre:g} s before to {spec.post:g} s after each event; baseline {spec.baseline[0]:g} to "
        f"{spec.baseline[1]:g} s; response {spec.response[0]:g} to {spec.response[1]:g} s")
    add(f"Signal: {_SIGNAL_TEXT.get(spec.signal, spec.signal)}"
        + (f"; smoothed with a {spec.smooth_seconds:g} s moving average" if spec.smooth_seconds > 0 else "; not smoothed")
        + f"; motion correction {spec.regression_method.upper()}")
    add("Measures: " + ", ".join(f"{METRIC_LABELS[m]} ({measure_units(m, spec.signal)})" for m in spec.metrics)
        + f"; peak = {_DIRECTION_TEXT[spec.peak_direction]}; decay = time to fall to {spec.decay_fraction:.0%} of the peak")
    add("")

    trials = results.trials
    add("TRIALS")
    if len(trials):
        table = trials.pivot_table(index="subject", columns="marker", values="trial", aggfunc="count", fill_value=0)
        table = table.reindex(index=[s for s in subjects if s in table.index],
                              columns=[m for m in spec.markers if m in table.columns])
        overlap = trials.groupby("marker")["overlap"].mean()
        add("  Trials per subject (all with a full window inside the recording):")
        add("\n".join("  " + line for line in table.to_string().splitlines()))
        add("  Share of trials whose window shares samples with a neighbouring trial's: "
            + ", ".join(f"{m} {overlap[m]:.0%}" for m in spec.markers if m in overlap.index))
        skipped = {m: n for m, n in results.excluded.items() if n}
        if skipped:
            add("  Events left out because their window is not fully inside the recording: "
                + ", ".join(f"{m} {n}" for m, n in skipped.items()))
    else:
        add("  No trials.")
    add("")

    add("RESULTS")
    for measure in spec.metrics:
        label = METRIC_LABELS[measure]
        add("")
        add(f"{label} ({measure_units(measure, spec.signal)})")
        add("-" * (len(label) + len(measure_units(measure, spec.signal)) + 3))
        d = results.descriptives
        d = d[d["measure"] == measure] if len(d) else d
        add("  Mean of the subject means, by marker:")
        if len(d):
            d = d.assign(**{c: [_num(v) for v in d[c]] for c in ("mean_of_subject_means", "sem", "ci_low", "ci_high")})
        add(_table(d, ["marker", "n_subjects", "n_trials", "mean_of_subject_means", "sem", "ci_low", "ci_high"],
                   ["marker", "subjects", "trials", "mean", "SEM", "95% CI low", "95% CI high"], indent=4).rstrip("\n"))
        for basis, title in (("trials", "Trial-level mixed model"), ("means", "Check on subject means")):
            m = results.models[(results.models["measure"] == measure) & (results.models["basis"] == basis)] if len(results.models) else results.models
            add("")
            if not len(m) or m.iloc[0]["status"] != "fitted":
                add(f"  {title}: not fitted (see the notes below).")
                continue
            info = m.iloc[0]
            add(f"  {title}: {info['formula']}, random effects: {info['random_effects']}; "
                f"{int(info['n_observations']):,} values from {int(info['n_subjects'])} subjects")
            om = results.omnibus[(results.omnibus["measure"] == measure) & (results.omnibus["basis"] == basis)] if len(results.omnibus) else results.omnibus
            if len(om):
                o = om.iloc[0]
                text = f"    Effect of marker: F({int(o['df1'])}, {_num(o['df2'])}) = {_num(o['F'])}, {_peq(o['p_F'])}"
                add(text + f"  (Wald chi2({int(o['df1'])}) = {_num(o['chi2'])}, {_peq(o['p_chi2'])})")
            fe = results.fixed_effects[(results.fixed_effects["measure"] == measure) & (results.fixed_effects["basis"] == basis)] if len(results.fixed_effects) else results.fixed_effects
            if int(info["n_markers"]) == 1 and len(fe):
                r = fe.iloc[0]
                if np.isfinite(r["p_t"]):
                    add(f"    {r['term']} against zero: estimate {_num(r['estimate'])}, 95% CI [{_num(r['ci_low'])}, "
                        f"{_num(r['ci_high'])}], t({_num(r['df'])}) = {_num(r['t'])}, {_peq(r['p_t'])}")
                else:
                    add(f"    {r['term']}: {_num(r['estimate'])}, 95% CI [{_num(r['ci_low'])}, {_num(r['ci_high'])}] "
                        "(not tested against zero: there is no null value for this measure)")
            pw = results.pairwise[(results.pairwise["measure"] == measure) & (results.pairwise["basis"] == basis)] if len(results.pairwise) else results.pairwise
            if len(pw):
                add(f"    Pairwise comparisons, p adjusted by {CORRECTION_LABELS.get(spec.correction, spec.correction)}"
                    f" (significant at {spec.alpha:g}: marked *):")
                shown = pw.assign(pair=pw["marker_a"] + " - " + pw["marker_b"],
                                  flag=np.where(pw["significant"], "*", ""),
                                  ci=[f"[{_num(a)}, {_num(b)}]" for a, b in zip(pw["ci_low"], pw["ci_high"])],
                                  tt=[f"t({_num(df)}) = {_num(t)}" for df, t in zip(pw["df"], pw["t"])],
                                  p=[_p(v) for v in pw["p_t"]], padj=[_p(v) for v in pw["p_adjusted"]],
                                  diff=[_num(v) for v in pw["difference"]])
                add(_table(shown, ["pair", "diff", "ci", "tt", "p", "padj", "flag"],
                           ["comparison", "difference", "95% CI", "statistic", "p", "p adjusted", ""]).rstrip("\n"))
            var = results.variance[(results.variance["measure"] == measure) & (results.variance["basis"] == basis)] if len(results.variance) else results.variance
            if len(var):
                add("    Variance: " + "; ".join(f"{r['component']} {_num(r['variance'])} ({r['share_of_total']:.0%})"
                                                  for _, r in var.iterrows()))
    add("")
    if results.notes:
        add("NOTES")
        for n in results.notes:
            add(f"  [{n['level']}] {n['text']}")
        add("")
    add("METHODS (a starting point to adapt)")
    add(_methods(results))
    return "\n".join(out) + "\n"


def _methods(results):
    spec = results.spec
    sw = results.software
    k = len(spec.markers)
    text = [
        f"Photometry recordings from {len(spec.subjects)} subjects were processed as dF/F (the isosbestic channel "
        f"regressed out of the signal channel by {spec.regression_method.upper()} regression, photobleaching removed with a "
        "double-exponential baseline, low-pass filtered at 5 Hz)"
        + (f" and smoothed with a {spec.smooth_seconds:g} s moving average." if spec.smooth_seconds > 0 else "."),
        f"For every occurrence of each marker ({', '.join(spec.markers)}) a window from {spec.pre:g} s before to "
        f"{spec.post:g} s after the event was taken; events whose window extended beyond the recording were excluded. "
        f"Each trial was baseline-corrected with the mean of {spec.baseline[0]:g} to {spec.baseline[1]:g} s"
        + (" and divided by the baseline SD" if spec.signal == "zscore" else "")
        + " (baseline statistics winsorised at 5 robust SDs).",
        f"In the response window ({spec.response[0]:g} to {spec.response[1]:g} s) the following were computed per trial: "
        + ", ".join(METRIC_LABELS[m] if METRIC_LABELS[m].isupper() else METRIC_LABELS[m].lower() for m in spec.metrics)
        + f" (AUC by the trapezoidal rule; peak = {_DIRECTION_TEXT[spec.peak_direction]}; "
        f"decay time = time from the peak to {spec.decay_fraction:.0%} of it, by linear interpolation).",
    ]
    if k >= 2:
        text.append(
            "Each measure was analysed with a linear mixed-effects model (statsmodels "
            f"{sw.get('statsmodels', '')} MixedLM, restricted maximum likelihood) with marker as a fixed effect and random "
            "intercepts for subject and for subject x marker. Tests use subject-based degrees of freedom (n - 1 for "
            "a contrast between two markers, (k - 1)(n - 1) for the omnibus F test of marker). All pairwise comparisons "
            f"between markers were corrected with {CORRECTION_LABELS.get(spec.correction, spec.correction)} "
            f"(alpha = {spec.alpha:g}). As a check that does not assume independent trials, each analysis was repeated on "
            "subject x marker means with subject as a random effect.")
    else:
        text.append(
            "Each measure was summarised with a linear mixed-effects model with a random intercept for subject "
            f"(statsmodels {sw.get('statsmodels', '')} MixedLM, REML). The AUC and mean amplitude were tested against zero "
            "with t tests on the model's intercept using n - 1 subject-based degrees of freedom; the peak, latency and decay "
            "time have no null value to test against and are reported as estimates with 95% intervals.")
    return "\n".join("  " + line for line in "\n\n".join(text).splitlines())
