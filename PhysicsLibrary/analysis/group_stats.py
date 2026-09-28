"""
analysis/group_stats.py
-----------------------
The statistics of group analysis, on the trial table group_trials.extract_group_trials builds:
descriptives, linear mixed-effects models, and pairwise comparisons between markers.

For each measure, twice:

  basis "trials"  every trial is a row. Two or more markers: measure ~ marker with a random
                  intercept for subject and one for each subject x marker cell (statsmodels
                  MixedLM, REML). One marker: measure ~ 1 with a random intercept for subject.
  basis "means"   one row per subject x marker (the mean of that cell's trials): measure ~ marker
                  with a random intercept for subject, which for a balanced design is the classical
                  repeated-measures analysis. It does not rely on trials being independent, so it is
                  the check on the first.

Why the subject x marker term: trials of one subject and marker share an effect of their own, and
neighbouring trials share signal when their windows overlap. A model without that term counts every
trial as independent evidence about a difference between markers. Simulated cohorts (three markers, 40
trials per subject and marker, 6, 10 and 20 subjects; 18 settings of 300 runs each, with and without a
subject x marker effect and with trials autocorrelated at 0.8) with no true difference between markers:
a random-intercept-only model called it significant in 65-93% of runs whenever trials were correlated
within a cell (4-8% when they were not), at a nominal 5%; the model used here in 1-8% of runs (4.8% on
average), as did repeated-measures ANOVA on the cell means (5.2%). With true differences (0, 0.5 and 1.0
residual SDs) and 10 subjects it found them in 95% of runs.

p-values. statsmodels gives Wald z tests, which assume many subjects. Alongside them (p_z) every test
also gets a small-sample version (p_t) that uses degrees of freedom from the subjects, not the trials:
n - 1 for a contrast between two markers (subjects that have both), like a paired test; n - 1 for a
mean against zero; and, for the omnibus test of a marker effect, an F with (k - 1, (k - 1)(n - 1)) as in
repeated-measures ANOVA. For balanced data these equal the paired t-test on the cell means. p_t is what
the pairwise comparisons are corrected on and what is flagged as significant.

One marker: there is nothing to compare it with. AUC and mean amplitude are tested against zero (a
baseline-corrected trace has mean zero when there is no response). The peak (the extreme of a noisy
window is never zero without a response), latency and decay time have no zero to test against, so they
are reported as estimates with 95% intervals only.
"""

import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
import statsmodels
import statsmodels.formula.api as smf
from patsy import dmatrix
from statsmodels.stats.multitest import multipletests

from ..progress import Plan
from .group import CORRECTION_LABELS, METRIC_LABELS

TESTABLE_AGAINST_ZERO = ("auc", "mean")

FRAME_NAMES = ("descriptives", "subject_means", "models", "fixed_effects", "omnibus", "variance",
               "estimated_means", "pairwise", "diagnostics", "random_effects")


@dataclass
class GroupResults:
    """Everything a group analysis produced. The DataFrames are the tables; `notes` are the things worth
    knowing (dropped markers, boundary fits, undefined measures) as {"level", "measure", "text"}."""

    spec: object
    trials: pd.DataFrame
    traces: dict = field(default_factory=dict)
    trace_grid: object = None
    excluded: dict = field(default_factory=dict)
    descriptives: pd.DataFrame = None
    subject_means: pd.DataFrame = None
    models: pd.DataFrame = None
    fixed_effects: pd.DataFrame = None
    omnibus: pd.DataFrame = None
    variance: pd.DataFrame = None
    estimated_means: pd.DataFrame = None
    pairwise: pd.DataFrame = None
    diagnostics: pd.DataFrame = None
    random_effects: pd.DataFrame = None
    notes: list = field(default_factory=list)
    software: dict = field(default_factory=dict)
    files: list = field(default_factory=list)          # paths written by group_report.write_group_results

    def frames(self):
        """{name: DataFrame} of every table, the trial table first."""
        out = {"trials": self.trials}
        for name in FRAME_NAMES:
            out[name] = getattr(self, name)
        return out


# ---- descriptives -------------------------------------------------------------------------------
def subject_means(trials, measures, markers):
    """Mean of each measure over the valid trials of every subject x marker cell (empty values skipped;
    the decay time, for one, can be empty). Long format: subject, marker, measure, n, mean."""
    valid = trials[trials["valid"].astype(bool)] if len(trials) else trials
    rows = []
    for measure in measures:
        if not len(valid):
            continue
        sub = valid[valid[measure].notna()]
        for (subject, marker), vals in sub.groupby(["subject", "marker"], sort=False)[measure]:
            rows.append((subject, marker, measure, int(len(vals)), float(vals.mean())))
    out = pd.DataFrame(rows, columns=["subject", "marker", "measure", "n", "mean"])
    return _sorted(out, measures, markers)


def _sorted(df, measures=None, markers=None):
    """Rows ordered by measure, marker (both in the order given), then subject; the helper keys are dropped again."""
    if not len(df):
        return df
    out, keys = df.copy(), []
    if measures is not None and "measure" in out:
        out["_m"] = out["measure"].map({m: i for i, m in enumerate(measures)})
        keys.append("_m")
    if markers is not None and "marker" in out:
        out["_k"] = out["marker"].map({m: i for i, m in enumerate(markers)})
        keys.append("_k")
    if "subject" in out:
        out["_s"] = out["subject"].astype(str)
        keys.append("_s")
    return out.sort_values(keys, kind="stable").drop(columns=keys).reset_index(drop=True)


def descriptives(trials, means, measures, markers):
    """Per measure and marker: how many subjects and trials, the mean and spread of the subject means (with the
    standard error and a 95% interval across subjects), and the mean and SD over all trials."""
    rows = []
    valid = trials[trials["valid"].astype(bool)] if len(trials) else trials
    for measure in measures:
        for marker in markers:
            cell = means[(means["measure"] == measure) & (means["marker"] == marker)]
            if not len(cell):
                continue
            vals = cell["mean"].to_numpy(float)
            n = len(vals)
            sd = float(vals.std(ddof=1)) if n > 1 else float("nan")
            sem = sd / math.sqrt(n) if n > 1 else float("nan")
            half = st.t.ppf(0.975, n - 1) * sem if n > 1 else float("nan")
            tr = valid[(valid["marker"] == marker)][measure].dropna().to_numpy(float)
            rows.append({"measure": measure, "marker": marker, "n_subjects": n, "n_trials": int(cell["n"].sum()),
                         "mean_of_subject_means": float(vals.mean()), "sd_of_subject_means": sd, "sem": sem,
                         "ci_low": float(vals.mean() - half), "ci_high": float(vals.mean() + half),
                         "median_of_subject_means": float(np.median(vals)),
                         "trial_mean": float(tr.mean()) if len(tr) else float("nan"),
                         "trial_sd": float(tr.std(ddof=1)) if len(tr) > 1 else float("nan")})
    return pd.DataFrame(rows)


# ---- one model ----------------------------------------------------------------------------------
# statsmodels' default optimizers (BFGS, L-BFGS, CG) often stop short of the REML optimum when a variance is
# near zero: on simulated cohorts they reported non-convergence in up to 62% of fits, and landed at a worse
# likelihood, which changes the standard errors. Powell's method (derivative-free) converged every time and
# reached the best likelihood found (300 fits), at about 50 ms a fit. It has no bounds, though, and with an
# unbalanced design (a subject without one of the markers) it can step onto a singular covariance and raise;
# the bounded gradient methods are the fallback for that.
OPTIMIZERS = (["powell", "nm"], ["lbfgs", "bfgs", "cg"])


def _fit_mixed(d, markers, cell_effect):
    """Fit the mixed model to d (columns y, subject, marker). Returns (result, None), or (None, reason) if it could
    not be fitted. statsmodels' own warnings (a variance at the boundary, an optimizer retrying) are swallowed: what
    they mean for the person is reported from the fitted variances instead."""
    k = len(markers)
    kwargs = {"groups": "subject"}
    if k >= 2 and cell_effect:
        kwargs.update(re_formula="1", vc_formula={"cell": "0 + C(marker)"})
    formula = "y ~ 1" if k == 1 else "y ~ C(marker)"
    problem = "the model did not converge"
    for methods in OPTIMIZERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = smf.mixedlm(formula, d, **kwargs).fit(reml=True, method=methods)
            except Exception as e:                           # singular matrices, non-finite likelihoods, ...
                problem = f"{type(e).__name__}: {e}"
                continue
        se = np.sqrt(np.diag(res.cov_params().loc[res.fe_params.index, res.fe_params.index].to_numpy(float)))
        if not res.converged or not np.all(np.isfinite(se)) or not np.all(np.isfinite(res.fe_params.to_numpy(float))):
            problem = "the model did not converge"
            continue
        return res, None
    return None, problem


def _contrast(res, row_a, row_b=None):
    """Estimate and standard error of a linear combination of the fixed effects (row_a, or row_a - row_b)."""
    beta = res.fe_params.to_numpy(float)
    cov = res.cov_params().loc[res.fe_params.index, res.fe_params.index].to_numpy(float)
    L = np.asarray(row_a, float) - (np.asarray(row_b, float) if row_b is not None else 0.0)
    return float(L @ beta), float(math.sqrt(max(L @ cov @ L, 0.0)))


def _tests(est, se, df):
    """z and t statistics with their two-sided p-values and the 95% interval from t (df may be NaN)."""
    z = est / se if se > 0 else (0.0 if est == 0 else math.copysign(math.inf, est))
    p_z = float(2 * st.norm.sf(abs(z)))
    if df is None or not np.isfinite(df) or df < 1:
        return {"z": z, "p_z": p_z, "df": float("nan"), "t": float("nan"), "p_t": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    p_t = float(2 * st.t.sf(abs(z), df))
    half = st.t.ppf(0.975, df) * se
    return {"z": z, "p_z": p_z, "df": float(df), "t": z, "p_t": p_t, "ci_low": est - half, "ci_high": est + half}


def _design_rows(res, markers):
    """The fixed-effects design row for each marker, aligned with res.fe_params."""
    if len(markers) == 1:
        return {markers[0]: np.ones(1)}
    info = res.model.data.design_info
    rows = {}
    for m in markers:
        frame = pd.DataFrame({"marker": pd.Categorical([m], categories=markers)})
        rows[m] = np.asarray(dmatrix(info, frame, return_type="dataframe"), float)[0]
    return rows


def _variances(res, k, cell_effect):
    out = [("subject", float(res.cov_re.iloc[0, 0]))]
    if k >= 2 and cell_effect:
        out.append(("subject x marker", float(np.atleast_1d(res.vcomp)[0])))
    out.append(("residual", float(res.scale)))
    return out


def analyse_measure(d, measure, markers, basis, correction="holm", alpha=0.05):
    """
    One measure on one basis. `d` has columns y, subject, marker (markers not in `markers`, and NaN, already gone).
    Returns a dict of small DataFrames (fixed_effects, omnibus, variance, estimated_means, pairwise, diagnostics,
    random_effects), a `model` dict describing the fit, and `notes`; or model["status"] != "fitted" when it could
    not be fitted, with the reason in notes.
    """
    label = METRIC_LABELS.get(measure, measure)
    notes = []
    empty = {"fixed_effects": [], "omnibus": [], "variance": [], "estimated_means": [], "pairwise": [],
             "diagnostics": pd.DataFrame(), "random_effects": []}

    def failed(text, level="warning"):
        notes.append({"level": level, "measure": measure, "text": f"{label} ({basis}): {text}"})
        return {**empty, "model": {"measure": measure, "basis": basis, "status": "not fitted"}, "notes": notes}

    counts = d.groupby("marker", observed=True)["subject"].nunique()
    kept = [m for m in markers if counts.get(m, 0) >= 2]
    for m in markers:
        if m not in kept:
            notes.append({"level": "warning", "measure": measure,
                          "text": f"{label} ({basis}): '{m}' has values for fewer than 2 subjects and was left out."})
    if not kept:
        return failed("no marker has values for two or more subjects.")
    d = d[d["marker"].isin(kept)].copy()
    d["marker"] = pd.Categorical(d["marker"], categories=kept)
    d["subject"] = d["subject"].astype(str)
    if d["y"].nunique() < 2:
        return failed("every value is identical, so there is nothing to model.")
    k = len(kept)
    n_subjects = int(d["subject"].nunique())
    cell_effect = basis == "trials"

    res, problem = _fit_mixed(d, kept, cell_effect)
    if res is None:
        return failed(problem + ("; the subject-means analysis does not depend on it." if basis == "trials" else "."))

    rows = _design_rows(res, kept)
    per_subject = {m: set(d.loc[d["marker"] == m, "subject"]) for m in kept}
    variances = _variances(res, k, cell_effect)
    total_var = sum(v for _, v in variances)
    total_sd = math.sqrt(total_var)
    at_zero = [name for name, v in variances if name != "residual" and v <= 1e-6 * total_var]
    if at_zero:
        notes.append({"level": "info", "measure": measure,
                      "text": f"{label} ({basis}): the {' and '.join(at_zero)} variance was estimated at zero (the data show "
                              "none beyond the residual), so the model reduces to a simpler one."})

    fixed, omnibus, emm, pair = [], [], [], []
    tested = k >= 2 or measure in TESTABLE_AGAINST_ZERO
    if k == 1:
        est, se = _contrast(res, rows[kept[0]])
        df = n_subjects - 1
        t = _tests(est, se, df)
        if not tested:
            t.update(z=float("nan"), p_z=float("nan"), t=float("nan"), p_t=float("nan"))
        fixed.append({"term": f"mean of '{kept[0]}'" + ("" if tested else " (estimate only)"), "estimate": est, "se": se, **t})
        emm.append({"marker": kept[0], "mean": est, "se": se, "df": t["df"], "ci_low": t["ci_low"], "ci_high": t["ci_high"]})
    else:
        ref = kept[0]
        names = [f"mean of '{ref}'"] + [f"'{m}' minus '{ref}'" for m in kept[1:]]
        beta_rows = [rows[ref]] + [rows[m] - rows[ref] for m in kept[1:]]
        for name, L, m in zip(names, beta_rows, [ref] + kept[1:]):
            est, se = _contrast(res, L)
            n_pair = len(per_subject[ref]) if m == ref else len(per_subject[ref] & per_subject[m])
            fixed.append({"term": name, "estimate": est, "se": se, **_tests(est, se, n_pair - 1)})
        for m in kept:
            est, se = _contrast(res, rows[m])
            t = _tests(est, se, len(per_subject[m]) - 1)
            emm.append({"marker": m, "mean": est, "se": se, "df": t["df"], "ci_low": t["ci_low"], "ci_high": t["ci_high"]})

        # Wald test that every marker coefficient is zero (the Intercept is the first fixed effect).
        beta = res.fe_params.to_numpy(float)[1:]
        cov = res.cov_params().loc[res.fe_params.index, res.fe_params.index].to_numpy(float)[1:, 1:]
        chi2 = float(beta @ np.linalg.solve(cov, beta))
        complete = set.intersection(*per_subject.values())
        df2 = (k - 1) * (len(complete) - 1) if len(complete) >= 2 else float("nan")
        omnibus.append({"test": "effect of marker", "chi2": chi2, "df1": k - 1, "p_chi2": float(st.chi2.sf(chi2, k - 1)),
                        "F": chi2 / (k - 1), "df2": df2,
                        "p_F": float(st.f.sf(chi2 / (k - 1), k - 1, df2)) if np.isfinite(df2) else float("nan"),
                        "n_subjects_with_all_markers": len(complete)})

        for a, b in itertools.combinations(kept, 2):
            est, se = _contrast(res, rows[a], rows[b])
            n_pair = len(per_subject[a] & per_subject[b])
            t = _tests(est, se, n_pair - 1)
            pair.append({"marker_a": a, "marker_b": b, "difference": est, "se": se, **t,
                         "std_difference": est / total_sd if total_sd > 0 else float("nan"),
                         "n_subjects_with_both": n_pair})
        if pair:
            p = np.array([r["p_t"] for r in pair], float)
            adj = np.full(len(p), np.nan)
            ok = np.isfinite(p)
            if ok.any():
                adj[ok] = multipletests(p[ok], alpha=alpha, method=correction)[1]
            for r, a_ in zip(pair, adj):
                r["p_adjusted"] = float(a_)
                r["significant"] = bool(a_ < alpha) if np.isfinite(a_) else False

    var_rows = [{"component": name, "variance": v, "sd": math.sqrt(v),
                 "share_of_total": v / sum(x for _, x in variances) if sum(x for _, x in variances) > 0 else float("nan")}
                for name, v in variances]
    resid = pd.DataFrame({"subject": d["subject"].to_numpy(), "marker": d["marker"].astype(str).to_numpy(),
                          "observed": d["y"].to_numpy(float), "fitted": np.asarray(res.fittedvalues, float),
                          "resid": np.asarray(res.resid, float)}) if basis == "trials" else pd.DataFrame()
    blups = [{"subject": str(s), "effect": float(np.asarray(v, float).ravel()[0])} for s, v in res.random_effects.items()]
    model = {"measure": measure, "basis": basis, "status": "fitted",
             "formula": "measure ~ 1" if k == 1 else "measure ~ marker",
             "random_effects": "subject" + (" + subject x marker" if k >= 2 and cell_effect else ""),
             "n_observations": int(len(d)), "n_subjects": n_subjects, "n_markers": k, "markers": "; ".join(kept),
             "tested_against_zero": bool(k == 1 and tested), "converged": bool(res.converged)}
    return {"model": model, "fixed_effects": fixed, "omnibus": omnibus, "variance": var_rows,
            "estimated_means": emm, "pairwise": pair, "diagnostics": resid, "random_effects": blups, "notes": notes}


# ---- the whole analysis -------------------------------------------------------------------------
def fit_group_models(trials, spec, progress=None, traces=None, trace_grid=None, excluded=None):
    """
    Descriptives, mixed models and pairwise comparisons for every measure in spec.metrics, on both bases.

    Parameters
    ----------
    trials : DataFrame
        The trial table (group_trials.TRIAL_COLUMNS), all subjects together.
    spec : GroupSpec
        Uses markers, metrics, correction and alpha.
    progress : None, True or callable(fraction, message)
        Progress reporting (see PhysicsLibrary.progress), one step per measure.
    traces, trace_grid, excluded
        Carried into the result unchanged (mean traces per subject and marker, their time axis, and the
        events left out for not having a full window) for the plots and the report.

    Returns
    -------
    GroupResults
    """
    markers = list(spec.markers)
    measures = list(spec.metrics)
    means = subject_means(trials, measures, markers)
    plan = Plan(progress, [(m, 1) for m in measures])
    notes, models, fixed, omni, var, emm, pair, diag, blup = [], [], [], [], [], [], [], [], []
    valid = trials[trials["valid"].astype(bool)] if len(trials) else trials

    for measure in measures:
        plan.begin(measure, f"Fitting models: {METRIC_LABELS.get(measure, measure)}")
        n_missing = int(valid[measure].isna().sum()) if len(valid) else 0
        if n_missing:
            notes.append({"level": "info", "measure": measure,
                          "text": f"{METRIC_LABELS.get(measure, measure)}: {n_missing} of {len(valid)} valid trials have "
                                  "no value (the trace never fell to the decay level, or there was no peak to fall from) "
                                  "and are left out of its models."})
        for basis in ("trials", "means"):
            if basis == "trials":
                d = valid[valid[measure].notna() & valid["marker"].isin(markers)][["subject", "marker", measure]] if len(valid) else pd.DataFrame(columns=["subject", "marker", "y"])
                d = d.rename(columns={measure: "y"}).reset_index(drop=True)
            else:
                cell = means[means["measure"] == measure]
                d = cell.rename(columns={"mean": "y"})[["subject", "marker", "y"]].reset_index(drop=True)
            out = analyse_measure(d, measure, markers, basis, spec.correction, spec.alpha)
            notes.extend(out["notes"])
            models.append(out["model"])
            for store, key in ((fixed, "fixed_effects"), (omni, "omnibus"), (var, "variance"), (emm, "estimated_means"),
                               (pair, "pairwise"), (blup, "random_effects")):
                store.extend({"measure": measure, "basis": basis, **r} for r in out[key])
            if len(out["diagnostics"]):
                diag.append(out["diagnostics"].assign(measure=measure))
    plan.done()
    for measure in measures:
        label = METRIC_LABELS.get(measure, measure)
        p = {}
        for r in omni:
            if r["measure"] == measure and np.isfinite(r["p_F"]):
                p[r["basis"]] = r["p_F"]
        if not p:                                                    # one marker: the test against zero instead
            for r in fixed:
                if r["measure"] == measure and np.isfinite(r["p_t"]):
                    p[r["basis"]] = r["p_t"]
        if len(p) == 2 and (p["trials"] < spec.alpha) != (p["means"] < spec.alpha):
            notes.append({"level": "warning", "measure": measure,
                          "text": f"{label}: the trial-level model (p = {p['trials']:.3g}) and the check on subject means "
                                  f"(p = {p['means']:.3g}) fall on different sides of {spec.alpha:g}. Trials from one subject "
                                  "are not independent evidence and subjects are what the tests count, so look at the "
                                  "subject means (subject_means.csv); the more cautious of the two is the safer one to report."})
    if len(trials) and len(valid) < len(trials):
        notes.append({"level": "warning", "measure": "",
                      "text": f"{len(trials) - len(valid)} of {len(trials)} trials could not be measured and are left out "
                              "(see the 'reason' column of the trial table)."})

    def frame(rows, cols=None):
        df = pd.DataFrame(rows)
        return df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]] if cols and len(df) else df

    results = GroupResults(
        spec=spec, trials=trials, traces=traces or {}, trace_grid=trace_grid, excluded=excluded or {},
        descriptives=descriptives(trials, means, measures, markers), subject_means=means,
        models=frame(models, ["measure", "basis", "status"]),
        fixed_effects=frame(fixed, ["measure", "basis", "term"]),
        omnibus=frame(omni, ["measure", "basis", "test"]),
        variance=frame(var, ["measure", "basis", "component"]),
        estimated_means=frame(emm, ["measure", "basis", "marker"]),
        pairwise=frame(pair, ["measure", "basis", "marker_a", "marker_b"]),
        diagnostics=pd.concat(diag, ignore_index=True) if diag else pd.DataFrame(),
        random_effects=frame(blup, ["measure", "basis", "subject"]),
        notes=notes,
        software={"statsmodels": statsmodels.__version__, "scipy": scipy.__version__,
                  "numpy": np.__version__, "pandas": pd.__version__},
    )
    return results
