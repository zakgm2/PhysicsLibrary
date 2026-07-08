"""
field_study_validation.py
----------------------------
Statistical validation for the paired-similarity metric computed by
text_field_study.py: for each field pair, is the same-subject
similarity meaningfully higher than chance, how big is that effect, is
it just a word-count artifact, how precise is the estimate, and how
much does it depend on any one subject?

run_validation_pipeline() is the single entry point (load -> embed ->
compute similarity -> validate, self-contained so it can run as its own
follow-up analysis without needing anything cached from a prior
run_field_study_pipeline() call); the stage functions below are also
exposed separately. Every function's docstring explains what the
statistic actually means, for interpreting/writing up results — not
just how to call it.
"""

import numpy as np
import pandas as pd

from .text_field_study import (
    load_field_study_folder,
    embed_text_fields,
    compute_paired_similarity,
    permutation_test_similarity,
)


def cohens_d(group1, group2):
    """
    Cohen's d: how far apart two distributions' means are, in units of
    their pooled standard deviation. Unlike a p-value (which just says
    "probably not the same"), this says how *big* the difference is —
    the conventional rough guide is ~0.2 small, ~0.5 medium, ~0.8 large,
    but those thresholds are a loose convention, not a hard rule.

    Here: group1 is typically the same-subject similarity values (one
    per subject), group2 the pooled null (chance-pairing) similarity
    values — d tells you how far the "real" pairing sits from chance
    pairing, in a way that doesn't depend on sample size the way a
    p-value does.

    Parameters
    ----------
    group1, group2 : array-like

    Returns
    -------
    float
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        # A standard deviation isn't defined from a single value — not
        # enough subjects/null draws yet for this to mean anything.
        return float("nan")
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return float((group1.mean() - group2.mean()) / pooled_std)


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg false discovery rate correction. Running several
    hypothesis tests (here: one permutation test per field pair) inflates
    the chance that at least one comes back "significant" by luck alone —
    this adjusts each p-value upward to control the *expected proportion*
    of false positives among the tests you call significant, which is
    less conservative (more power) than a stricter correction like
    Bonferroni while still accounting for doing multiple tests.

    The output is often called a "q-value": if you use q < 0.05 as your
    cutoff, you're accepting that on average 5% of the pairs you call
    significant across repeated studies like this one would be false
    positives — not that any individual result has a 5% chance of being
    wrong.

    Parameters
    ----------
    p_values : array-like

    Returns
    -------
    ndarray
        Corrected p-values (q-values), same order as the input.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    corrected = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity (a lower-ranked p-value's correction can never
    # end up smaller than a higher-ranked one's) via a reverse cumulative
    # minimum — the standard BH step-up procedure.
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)
    out = np.empty(n)
    out[order] = corrected
    return out


def wordcount_controlled_regression(df, field_a, field_b, sim_col):
    """
    OLS regression: same-subject similarity ~ wordcount_<field_a> +
    wordcount_<field_b>. Tests whether verbosity in either field predicts
    similarity, controlling for the other. If a word-count coefficient
    comes back significant (small p-value), that field's length is
    predicting similarity independent of actual content overlap — a
    "longer answers just look more similar" confound that would weaken a
    claim based on the similarity metric alone. If neither is
    significant, the observed similarity isn't just a length artifact.

    Requires the `statsmodels` package.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have wordcount_<field_a>, wordcount_<field_b>, and sim_col
        columns (i.e. called on the output of run_field_study_pipeline,
        or with those columns present some other way).
    field_a, field_b : str
    sim_col : str
        The similarity column to use as the outcome, e.g. 'sim_<name>'.

    Returns
    -------
    dict
        r_squared : float
            Proportion of variance in similarity explained by both word
            counts together — small values mean word count barely
            matters, which is the reassuring result for the confound
            check.
        coef_const, coef_wordcount_<field_a>, coef_wordcount_<field_b> : float
            Regression coefficients (the intercept, and how much
            similarity is predicted to change per additional word in
            each field).
        pvalue_const, pvalue_wordcount_<field_a>, pvalue_wordcount_<field_b> : float
            Whether each coefficient is distinguishable from zero.
        model : statsmodels RegressionResults
            The full fitted model, for anyone who wants more than the
            summary numbers above (e.g. confidence intervals, residual
            diagnostics).
    """
    import statsmodels.api as sm

    x = df[[f"wordcount_{field_a}", f"wordcount_{field_b}"]].astype(float)
    x = sm.add_constant(x)
    y = df[sim_col].astype(float)

    # 3 parameters (intercept + 2 word-count covariates) need at least 3
    # subjects to not be a singular/degenerate fit — with fewer, there's
    # no unique answer, so return NaNs instead of letting statsmodels
    # produce a nonsensical or all-NaN result silently.
    if len(df) < x.shape[1]:
        result = {"r_squared": float("nan"), "model": None}
        for name in x.columns:
            result[f"coef_{name}"] = float("nan")
            result[f"pvalue_{name}"] = float("nan")
        return result

    model = sm.OLS(y, x).fit()

    result = {"r_squared": float(model.rsquared), "model": model}
    for name in x.columns:
        result[f"coef_{name}"] = float(model.params[name])
        result[f"pvalue_{name}"] = float(model.pvalues[name])
    return result


def bootstrap_mean_ci(values, n_boot=1000, ci=0.95, rng_seed=None):
    """
    Bootstrap confidence interval on the mean: resample the subjects
    (with replacement, same sample size) n_boot times, take the mean
    each time, and report the [lower, upper] percentiles of that
    distribution of means. This doesn't assume the underlying similarity
    values are normally distributed (unlike a textbook mean ± 1.96*SE
    interval) — it just asks "how much would my estimate of the mean
    wobble if I'd happened to sample a slightly different set of
    subjects from the same population."

    A narrow interval means the estimate is precise (more subjects, or
    less subject-to-subject variability); a wide one means treat the
    point estimate cautiously — you don't have enough subjects yet to
    pin it down tightly.

    Parameters
    ----------
    values : array-like
        Per-subject values (e.g. same-subject similarity for one pair).
    n_boot : int
        Number of resamples.
    ci : float
        Confidence level, e.g. 0.95 for a 95% interval.
    rng_seed : int or None

    Returns
    -------
    dict
        mean, ci_lower, ci_upper, boot_means (the full resampled-means
        array, for anyone who wants to plot its distribution).
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = rng.choice(values, size=n, replace=True).mean()

    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return {
        "mean": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "boot_means": boot_means,
    }


def leave_one_out_sensitivity(values, ids=None):
    """
    Leave-one-out sensitivity check: recompute the mean with each subject
    dropped, one at a time, and flag any subject whose removal shifts the
    mean by more than one standard deviation of the resulting
    (n_subjects-long) distribution of leave-one-out means.

    This catches a single unusual subject quietly dominating your
    result — e.g. one participant with an oddly high or low similarity
    score dragging the whole-group mean toward it. A flagged subject
    isn't necessarily wrong or worth excluding — it's worth looking at
    that subject's actual data and deciding, not an automatic exclusion
    rule.

    Parameters
    ----------
    values : array-like
        Per-subject values (e.g. same-subject similarity for one pair).
    ids : array-like or None
        Subject identifiers matching values' order, for readable output.
        Defaults to positional indices if not given.

    Returns
    -------
    dict
        full_mean : float
        loo_means : ndarray
            The mean with each subject left out, in input order.
        shifts : ndarray
            full_mean minus each loo_means entry — how far removing that
            subject moved the mean.
        threshold : float
            One standard deviation of loo_means — the cutoff used to flag.
        flagged_ids : list
            Subject ids whose |shift| exceeds threshold.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if ids is None:
        ids = list(range(n))
    else:
        ids = list(ids)

    full_mean = float(values.mean()) if n > 0 else float("nan")
    if n < 2:
        # Nothing to leave out-and-compare with fewer than 2 subjects —
        # "does removing one subject change the mean" isn't meaningful
        # when there's only one (or zero) to begin with.
        return {
            "full_mean": full_mean,
            "loo_means": np.array([]),
            "shifts": np.array([]),
            "threshold": 0.0,
            "flagged_ids": [],
        }

    loo_means = np.array([np.delete(values, i).mean() for i in range(n)])
    shifts = full_mean - loo_means
    threshold = float(loo_means.std(ddof=1))
    flagged_mask = np.abs(shifts) > threshold

    return {
        "full_mean": full_mean,
        "loo_means": loo_means,
        "shifts": shifts,
        "threshold": threshold,
        "flagged_ids": [ids[i] for i in range(n) if flagged_mask[i]],
    }


def build_validation_summary(df, paired_fields, similarity_results, id_field="participant_id", n_boot=1000, rng_seed=None):
    """
    Folds every check above into one summary table, one row per field
    pair — see each column's meaning below (and the stage functions'
    own docstrings for the full explanation of each statistic).

    Parameters
    ----------
    df : pandas.DataFrame
        Needs wordcount_<field> columns for every field in paired_fields,
        and id_field if you want subject ids (not positional indices) in
        flagged_participant_ids.
    paired_fields : list of (field_a, field_b, pair_name)
    similarity_results : dict
        As returned by compute_paired_similarity, computed with these
        same paired_fields — reused rather than recomputed here, since
        recomputing similarity means recomputing the (expensive)
        embeddings.
    id_field : str
    n_boot : int
    rng_seed : int or None

    Returns
    -------
    pandas.DataFrame
        One row per pair:
        - pair, field_a, field_b : which fields this row covers
        - observed_mean_similarity : mean same-subject similarity
        - p_value : permutation-test p-value (is the observed mean
          higher than chance-pairing, one-tailed)
        - p_value_fdr : p_value after Benjamini-Hochberg correction
          across all pairs in this table — use this one, not the raw
          p_value, once you're looking at more than one pair
        - cohens_d : effect size (same-subject similarity vs. the pooled
          chance-pairing null), pooled-std units
        - wc_coef_a/wc_pvalue_a, wc_coef_b/wc_pvalue_b : word-count
          regression coefficient/p-value for field_a's and field_b's
          word count respectively, controlling for each other
          (wordcount_controlled_regression) — a significant coefficient
          here means part of the similarity effect may just be verbosity
        - regression_r_squared : how much of the similarity variance
          both word counts together explain (low = reassuring)
        - ci_lower, ci_upper : bootstrap 95% CI on observed_mean_similarity
        - n_flagged_loo : how many subjects were flagged by the
          leave-one-out check (0 is the reassuring result)
        - flagged_participant_ids : which ones, comma-separated
    """
    rows = []
    p_values = []
    for field_a, field_b, name in paired_fields:
        result = similarity_results[name]
        same = result["same_subject_similarity"]
        null_values = result["null_values"]
        null_shuffle_means = result["null_shuffle_means"]

        perm = permutation_test_similarity(same, null_shuffle_means)
        d = cohens_d(same, null_values)
        reg = wordcount_controlled_regression(df, field_a, field_b, f"sim_{name}")
        ci = bootstrap_mean_ci(same, n_boot=n_boot, rng_seed=rng_seed)
        ids = df[id_field].tolist() if id_field in df.columns else None
        loo = leave_one_out_sensitivity(same, ids=ids)

        rows.append({
            "pair": name,
            "field_a": field_a,
            "field_b": field_b,
            "observed_mean_similarity": perm["observed_mean_similarity"],
            "p_value": perm["p_value"],
            "cohens_d": d,
            "wc_coef_a": reg[f"coef_wordcount_{field_a}"],
            "wc_pvalue_a": reg[f"pvalue_wordcount_{field_a}"],
            "wc_coef_b": reg[f"coef_wordcount_{field_b}"],
            "wc_pvalue_b": reg[f"pvalue_wordcount_{field_b}"],
            "regression_r_squared": reg["r_squared"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "n_flagged_loo": len(loo["flagged_ids"]),
            "flagged_participant_ids": ",".join(str(x) for x in loo["flagged_ids"]),
        })
        p_values.append(perm["p_value"])

    p_fdr = benjamini_hochberg(np.array(p_values))
    for row, q in zip(rows, p_fdr):
        row["p_value_fdr"] = float(q)

    return pd.DataFrame(rows)


def run_validation_pipeline(folder_path, text_fields, paired_fields,
                             model_name="all-MiniLM-L6-v2", n_null=200,
                             n_boot=1000, rng_seed=None, file_glob="P-*.json",
                             id_field="participant_id"):
    """
    Full validation pipeline: load -> embed -> compute paired similarity
    -> build the validation summary table (see build_validation_summary
    for what each column means). Self-contained — recomputes embeddings
    rather than reusing a prior run_field_study_pipeline() call, so this
    can run as its own follow-up analysis.

    Parameters
    ----------
    folder_path : str
    text_fields : list of str
        Same meaning as in run_field_study_pipeline.
    paired_fields : list of (field_a, field_b, pair_name)
        Which fields to compare. Required here (unlike
        run_field_study_pipeline, where it's optional) — there's nothing
        to validate without at least one pair.
    model_name : str
    n_null : int
        Permutation shuffles for the null distribution.
    n_boot : int
        Bootstrap resamples for the confidence interval.
    rng_seed : int or None
    file_glob : str
    id_field : str
        Column to use as subject identifiers in the leave-one-out flags.

    Returns
    -------
    pandas.DataFrame
        See build_validation_summary.
    """
    if not paired_fields:
        raise ValueError("paired_fields is required — nothing to validate without at least one pair.")

    df = load_field_study_folder(folder_path, text_fields, file_glob=file_glob)
    fields_to_embed = sorted({f for field_a, field_b, _ in paired_fields for f in (field_a, field_b)})
    embeddings = embed_text_fields(df, fields_to_embed, model_name=model_name)
    similarity_results = compute_paired_similarity(embeddings, paired_fields, n_null=n_null, rng_seed=rng_seed)

    # build_validation_summary's regression step needs sim_<name> columns
    # on df, same as run_field_study_pipeline adds them.
    for field_a, field_b, name in paired_fields:
        df[f"sim_{name}"] = similarity_results[name]["same_subject_similarity"]

    return build_validation_summary(
        df, paired_fields, similarity_results,
        id_field=id_field, n_boot=n_boot, rng_seed=rng_seed,
    )
