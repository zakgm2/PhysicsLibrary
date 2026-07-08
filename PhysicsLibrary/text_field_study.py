"""
text_field_study.py
---------------------
Generic pipeline for a study where each subject produces one JSON file
with several free-text fields, and you want to compare specific pairs
of fields to each other (e.g. "does this answer track that one for the
same subject"). Domain-agnostic by design — field names and which pairs
to compare are supplied by the caller via arguments, not hardcoded
here, so this works for any text-field study, not one specific one.

run_field_study_pipeline() is the single entry point for running the
whole thing at once; the stage functions below are also exposed for
anyone who wants an intermediate result (embeddings, raw null
distributions) directly.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def peek_fields(folder_path, file_glob="P-*.json"):
    """
    Returns the field names (dict keys) found in the first file matching
    file_glob in folder_path, without loading the whole folder — lets a
    caller (e.g. a GUI) show a user their actual field names to assign
    to groups, rather than requiring them to already know/type them.

    Parameters
    ----------
    folder_path : str
    file_glob : str

    Returns
    -------
    list of str
    """
    paths = sorted(glob.glob(os.path.join(folder_path, file_glob)))
    if not paths:
        raise ValueError(f"No files matching '{file_glob}' found in {folder_path}")
    with open(paths[0], "r", encoding="utf-8") as fh:
        return list(json.load(fh).keys())


def load_field_study_folder(folder_path, text_fields, file_glob="P-*.json"):
    """
    Load every file matching file_glob in folder_path into a DataFrame
    (one row per subject) and add a wordcount_<field> column for each
    text field — a covariate for later analysis, not itself a feature.

    Parameters
    ----------
    folder_path : str
    text_fields : list of str
        Every free-text field to word-count.
    file_glob : str
        Filename pattern within folder_path, e.g. "P-*.json".

    Returns
    -------
    pandas.DataFrame
    """
    paths = sorted(glob.glob(os.path.join(folder_path, file_glob)))
    if not paths:
        raise ValueError(f"No files matching '{file_glob}' found in {folder_path}")

    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            records.append(json.load(fh))
    df = pd.DataFrame(records)

    for field in text_fields:
        text = df[field].fillna("") if field in df.columns else ""
        df[f"wordcount_{field}"] = text.apply(lambda s: len(str(s).split()))

    return df


def flag_low_quality(df, text_fields, min_words=5):
    """
    Flags (doesn't drop) rows with near-empty responses — noise, not
    signal, but the caller decides whether/how to exclude them, so
    nothing here is silently discarded.

    Adds a low_quality_<field> bool column per text field (True if that
    field's word count is below min_words) and an any_low_quality column
    (True if any field is flagged for that subject). Must be called
    after word counts exist (i.e. after load_field_study_folder).

    Parameters
    ----------
    df : pandas.DataFrame
        As returned by load_field_study_folder — must have
        wordcount_<field> columns for text_fields.
    text_fields : list of str
    min_words : int
        Fields with fewer words than this are flagged.

    Returns
    -------
    pandas.DataFrame
        The same df, with the new columns added.
    """
    flag_cols = []
    for field in text_fields:
        col = f"low_quality_{field}"
        df[col] = df[f"wordcount_{field}"] < min_words
        flag_cols.append(col)
    df["any_low_quality"] = df[flag_cols].any(axis=1)
    return df


def embed_text_fields(df, fields, model_name="all-MiniLM-L6-v2"):
    """
    Embed each of `fields` with a sentence-transformers model,
    L2-normalized (so a plain dot product below is the cosine similarity).
    Flat by field name — nothing about which fields get compared to
    which needs to be decided at embedding time, only when you call
    compute_delta_vector/compute_paired_similarity afterward.

    Parameters
    ----------
    df : pandas.DataFrame
        As returned by load_field_study_folder.
    fields : list of str
        Which fields to embed (typically every field referenced by your
        paired_fields/delta_pair — no need to embed fields you're not
        actually comparing).
    model_name : str

    Returns
    -------
    dict
        {field: (n_subjects, dim) ndarray, ...}
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = {}
    for field in fields:
        texts = df[field].fillna("").astype(str).tolist()
        embeddings[field] = np.asarray(model.encode(texts, normalize_embeddings=True))
    return embeddings


def compute_delta_vector(embeddings, field_from, field_to):
    """
    delta = vec(field_to) - vec(field_from), per subject, plus its
    magnitude — e.g. how much a subject's answer changed between two
    related fields.

    Parameters
    ----------
    embeddings : dict
        As returned by embed_text_fields.
    field_from, field_to : str

    Returns
    -------
    (delta_vectors, delta_magnitude) : (ndarray, ndarray)
    """
    vectors = embeddings[field_to] - embeddings[field_from]
    magnitude = np.linalg.norm(vectors, axis=1)
    return vectors, magnitude


def compute_paired_similarity(embeddings, paired_fields, n_null=200, rng_seed=None):
    """
    Cosine similarity between each pair of fields for the same subject,
    plus a null distribution built by repeatedly shuffling one field's
    vectors across subjects and recomputing similarity — the basis for a
    later permutation test on whether same-subject similarity is higher
    than chance pairing.

    Each shuffle is a full permutation across all subjects at once; all
    n_null * n_subjects values across every shuffle are pooled into one
    null distribution per pair (not per subject — "chance pairing" is a
    property of the pairing process, not of any one subject). A
    permutation occasionally maps a subject to themselves (identity
    mapping isn't excluded); for a reasonable sample size this is a
    negligible bias, not worth the extra complexity of a derangement.

    Parameters
    ----------
    embeddings : dict
        As returned by embed_text_fields.
    paired_fields : list of (field_a, field_b, pair_name)
        Which two fields to compare, and a name for that comparison.
    n_null : int
        Number of random shuffles to build the null distribution from.
    rng_seed : int or None

    Returns
    -------
    dict
        {pair_name: {
            'same_subject_similarity': (n_subjects,) ndarray,
            'null_mean': float,
            'null_std': float,
            'null_values': (n_null * n_subjects,) ndarray,
            'null_shuffle_means': (n_null,) ndarray,
                the mean similarity within each individual shuffle —
                this (not null_values) is the actual null distribution
                for a permutation test of the *mean* same-subject
                similarity; see permutation_test_similarity.
        }, ...}
    """
    rng = np.random.default_rng(rng_seed)
    results = {}
    for field_a, field_b, name in paired_fields:
        vecs_a = embeddings[field_a]
        vecs_b = embeddings[field_b]
        n = vecs_a.shape[0]

        same_subject_similarity = np.sum(vecs_a * vecs_b, axis=1)

        null_values = np.empty(n_null * n)
        null_shuffle_means = np.empty(n_null)
        for i in range(n_null):
            shuffled = vecs_b[rng.permutation(n)]
            shuffle_sims = np.sum(vecs_a * shuffled, axis=1)
            null_values[i * n:(i + 1) * n] = shuffle_sims
            null_shuffle_means[i] = shuffle_sims.mean()

        results[name] = {
            "same_subject_similarity": same_subject_similarity,
            "null_mean": float(null_values.mean()),
            "null_std": float(null_values.std()),
            "null_values": null_values,
            "null_shuffle_means": null_shuffle_means,
        }
    return results


def permutation_test_similarity(same_subject_similarity, null_shuffle_means):
    """
    Permutation test: is the observed mean same-subject similarity
    higher than expected under random pairing? Compares the observed
    mean against the distribution of per-shuffle means (not the pooled
    individual null_values — the test statistic is the mean, so its
    null distribution has to be built from the same statistic computed
    under each shuffle, not from individual chance-pairing similarities).

    One-tailed (is same-subject similarity higher than chance, not just
    different) — add-one smoothing avoids a p-value of exactly 0 for a
    finite number of shuffles.

    Parameters
    ----------
    same_subject_similarity : ndarray
        Per-subject observed similarity (one pair's worth).
    null_shuffle_means : ndarray
        Per-shuffle mean similarity, as returned by
        compute_paired_similarity.

    Returns
    -------
    dict
        observed_mean_similarity, p_value,
        effect_size (observed mean minus the null shuffle-mean
        distribution's own mean, in that distribution's std units),
        null_shuffle_mean, null_shuffle_std.
    """
    observed = float(np.mean(same_subject_similarity))
    n_null = len(null_shuffle_means)
    p_value = (np.sum(null_shuffle_means >= observed) + 1) / (n_null + 1)

    shuffle_mean = float(null_shuffle_means.mean())
    shuffle_std = float(null_shuffle_means.std())
    effect_size = (observed - shuffle_mean) / shuffle_std if shuffle_std > 0 else float("nan")

    return {
        "observed_mean_similarity": observed,
        "p_value": float(p_value),
        "effect_size": effect_size,
        "null_shuffle_mean": shuffle_mean,
        "null_shuffle_std": shuffle_std,
    }


def wordcount_confound_check(df, paired_fields):
    """
    Checks whether word count predicts similarity — a "more words = more
    similar" confound would show up as a significant positive
    correlation here, which would undercut any claim based on the
    similarity metric alone.

    Pearson correlation between each pair's combined word count
    (wordcount_<field_a> + wordcount_<field_b>) and its similarity
    column (sim_<name>) across subjects.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have wordcount_<field_a>, wordcount_<field_b>, and
        sim_<name> columns for each pair (i.e. called after
        run_field_study_pipeline's word-count and similarity steps).
    paired_fields : list of (field_a, field_b, pair_name)

    Returns
    -------
    dict
        {pair_name: {'r': float, 'p_value': float}, ...}
        Both are NaN if there are fewer than 2 subjects, or every
        subject has the same word count — a correlation genuinely isn't
        defined in either case, not something to error out over.
    """
    results = {}
    for field_a, field_b, name in paired_fields:
        wordcount = df[f"wordcount_{field_a}"] + df[f"wordcount_{field_b}"]
        similarity = df[f"sim_{name}"]
        if len(wordcount) < 2 or wordcount.nunique() < 2 or similarity.nunique() < 2:
            results[name] = {"r": float("nan"), "p_value": float("nan")}
            continue
        r, p_value = stats.pearsonr(wordcount, similarity)
        results[name] = {"r": float(r), "p_value": float(p_value)}
    return results


def run_field_study_pipeline(folder_path, text_fields, delta_pair=None, paired_fields=None,
                              model_name="all-MiniLM-L6-v2", n_null=200, rng_seed=None,
                              file_glob="P-*.json", min_words=5):
    """
    Full pipeline: load -> flag low-quality responses -> embed -> optional
    delta vector -> optional paired similarity (+ permutation test +
    word-count confound check), folded into one DataFrame (one row per
    subject). This is the single entry point for running the whole thing
    at once (e.g. as one background job); the stage functions above are
    available separately for anyone who wants an intermediate result
    (e.g. the raw null distributions, not carried into the DataFrame
    here — only their mean/std/permutation-test summary are).

    Parameters
    ----------
    folder_path : str
    text_fields : list of str
        Every free-text field to load/word-count.
    delta_pair : (field_from, field_to) or None
        If given, adds a delta_magnitude column:
        ||vec(field_to) - vec(field_from)||.
    paired_fields : list of (field_a, field_b, pair_name) or None
        If given, adds per pair: sim_<name>, null_mean_<name>,
        null_std_<name> (individual chance-pairing similarity stats),
        pvalue_<name>/effect_size_<name> (permutation test of whether
        mean same-subject similarity beats chance pairing), and
        wc_confound_r_<name>/wc_confound_p_<name> (does combined word
        count predict similarity — a possible "more words = more
        similar" confound). The test-level values (p-value, effect
        size, confound r) are the same for every row, same as
        null_mean_/null_std_ already were — one number per pair, not
        per subject.
    model_name : str
    n_null : int
    rng_seed : int or None
    file_glob : str
    min_words : int
        Fields with fewer words than this get flagged (not dropped) —
        see flag_low_quality.

    Returns
    -------
    pandas.DataFrame
    """
    df = load_field_study_folder(folder_path, text_fields, file_glob=file_glob)
    df = flag_low_quality(df, text_fields, min_words=min_words)

    # Only embed fields actually referenced by delta_pair/paired_fields —
    # embedding is the expensive step, no reason to run it over fields
    # nobody's comparing.
    fields_to_embed = set()
    if delta_pair is not None:
        fields_to_embed.update(delta_pair)
    for field_a, field_b, _ in (paired_fields or []):
        fields_to_embed.update([field_a, field_b])
    embeddings = embed_text_fields(df, sorted(fields_to_embed), model_name=model_name)

    if delta_pair is not None:
        field_from, field_to = delta_pair
        _, magnitude = compute_delta_vector(embeddings, field_from, field_to)
        df["delta_magnitude"] = magnitude

    if paired_fields:
        results = compute_paired_similarity(embeddings, paired_fields, n_null=n_null, rng_seed=rng_seed)
        for name, result in results.items():
            df[f"sim_{name}"] = result["same_subject_similarity"]
            df[f"null_mean_{name}"] = result["null_mean"]
            df[f"null_std_{name}"] = result["null_std"]

            test = permutation_test_similarity(
                result["same_subject_similarity"], result["null_shuffle_means"]
            )
            df[f"pvalue_{name}"] = test["p_value"]
            df[f"effect_size_{name}"] = test["effect_size"]

        confound = wordcount_confound_check(df, paired_fields)
        for name, result in confound.items():
            df[f"wc_confound_r_{name}"] = result["r"]
            df[f"wc_confound_p_{name}"] = result["p_value"]

    return df
