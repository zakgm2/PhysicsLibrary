"""Statistics used to interpret the text-field-study similarity metric: effect size,
multiple-comparison correction, bootstrap CI, leave-one-out, the permutation test
and the word-count confound checks. Where possible each result is compared with an
INDEPENDENT implementation (statsmodels, scipy, plain least squares) or a hand
calculation, not with the library's own logic."""

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import stats

from PhysicsLibrary.field_study_validation import (
    benjamini_hochberg, bootstrap_mean_ci, cohens_d, leave_one_out_sensitivity,
    wordcount_controlled_regression,
)
from PhysicsLibrary.text_field_study import (
    compute_delta_vector, compute_paired_similarity, flag_low_quality,
    permutation_test_similarity, wordcount_confound_check,
)


def _unit_rows(n, dim, seed):
    v = np.random.default_rng(seed).normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestCohensD(unittest.TestCase):
    def test_hand_computed_value(self):
        # means 3 and 5, both variances 2.5 -> pooled sd sqrt(2.5): d = -2 / 1.5811 = -1.2649
        d = cohens_d([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
        self.assertAlmostEqual(d, -2 / np.sqrt(2.5), places=12)

    def test_sign_flips_with_the_argument_order_and_identical_groups_give_zero(self):
        a, b = [1.0, 2.0, 4.0, 7.0], [2.0, 3.0, 3.5, 9.0]
        self.assertAlmostEqual(cohens_d(a, b), -cohens_d(b, a), places=12)
        self.assertEqual(cohens_d(a, a), 0.0)

    def test_it_is_unchanged_by_shifting_and_rescaling_both_groups(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(1, 2, 50), rng.normal(0, 2, 60)
        self.assertAlmostEqual(cohens_d(a, b), cohens_d(7 * a + 3, 7 * b + 3), places=10)

    def test_recovers_a_known_effect_size(self):
        rng = np.random.default_rng(1)
        d = cohens_d(rng.normal(1.0, 1.0, 20000), rng.normal(0.0, 1.0, 20000))
        self.assertAlmostEqual(d, 1.0, delta=0.05)

    def test_undefined_cases_return_nan(self):
        self.assertTrue(np.isnan(cohens_d([1.0], [1.0, 2.0])))              # < 2 values in a group
        self.assertTrue(np.isnan(cohens_d([2.0, 2.0], [2.0, 2.0])))         # zero pooled variance


class TestBenjaminiHochberg(unittest.TestCase):
    def test_hand_computed_example(self):
        # p * n / rank = 0.05 for every entry here
        assert_allclose(benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05]), [0.05] * 5)

    def test_matches_statsmodels_and_scipy_on_a_textbook_example(self):
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
        from statsmodels.stats.multitest import multipletests
        assert_allclose(benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])
        if hasattr(stats, "false_discovery_control"):
            assert_allclose(benjamini_hochberg(p), stats.false_discovery_control(p, method="bh"))

    def test_matches_statsmodels_on_random_input_in_the_original_order(self):
        from statsmodels.stats.multitest import multipletests
        p = np.random.default_rng(2).uniform(0, 1, 200) ** 3           # plenty of small p-values
        assert_allclose(benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])

    def test_structural_properties(self):
        p = np.random.default_rng(3).uniform(0, 1, 50)
        q = benjamini_hochberg(p)
        self.assertTrue(np.all(q >= p - 1e-12))                            # correction never lowers a p-value
        self.assertTrue(np.all((q >= 0) & (q <= 1)))
        self.assertTrue(np.all(np.diff(q[np.argsort(p)]) >= -1e-12))       # monotone in p
        assert_allclose(benjamini_hochberg([0.5, 0.5, 0.5]), [0.5, 0.5, 0.5])   # ties


class TestBootstrapCI(unittest.TestCase):
    def setUp(self):
        self.values = np.random.default_rng(4).normal(0.0, 1.0, 400)

    def test_seeded_results_are_reproducible_and_the_mean_is_exact(self):
        a = bootstrap_mean_ci(self.values, n_boot=500, rng_seed=7)
        b = bootstrap_mean_ci(self.values, n_boot=500, rng_seed=7)
        self.assertEqual((a["ci_lower"], a["ci_upper"]), (b["ci_lower"], b["ci_upper"]))
        self.assertAlmostEqual(a["mean"], self.values.mean(), places=12)
        self.assertEqual(len(a["boot_means"]), 500)

    def test_interval_brackets_the_mean_and_has_the_textbook_width(self):
        r = bootstrap_mean_ci(self.values, n_boot=4000, ci=0.95, rng_seed=1)
        self.assertLess(r["ci_lower"], r["mean"])
        self.assertGreater(r["ci_upper"], r["mean"])
        expected_width = 2 * 1.96 * self.values.std(ddof=1) / np.sqrt(len(self.values))
        self.assertAlmostEqual((r["ci_upper"] - r["ci_lower"]) / expected_width, 1.0, delta=0.15)

    def test_higher_confidence_gives_a_wider_interval(self):
        narrow = bootstrap_mean_ci(self.values, n_boot=2000, ci=0.5, rng_seed=1)
        wide = bootstrap_mean_ci(self.values, n_boot=2000, ci=0.99, rng_seed=1)
        self.assertLess(narrow["ci_upper"] - narrow["ci_lower"], wide["ci_upper"] - wide["ci_lower"])

    def test_constant_values_give_a_degenerate_interval(self):
        r = bootstrap_mean_ci([3.0] * 10, n_boot=100, rng_seed=0)
        self.assertEqual((r["ci_lower"], r["mean"], r["ci_upper"]), (3.0, 3.0, 3.0))


class TestLeaveOneOut(unittest.TestCase):
    def test_hand_computed_case_flags_only_the_outlier(self):
        r = leave_one_out_sensitivity([1, 1, 1, 1, 10], ids=["a", "b", "c", "d", "e"])
        self.assertAlmostEqual(r["full_mean"], 2.8)
        assert_allclose(r["loo_means"], [3.25, 3.25, 3.25, 3.25, 1.0])       # e.g. (1+1+1+10)/4 = 3.25
        assert_allclose(r["shifts"], [-0.45, -0.45, -0.45, -0.45, 1.8])
        self.assertAlmostEqual(r["threshold"], np.std([3.25] * 4 + [1.0], ddof=1))
        self.assertEqual(r["flagged_ids"], ["e"])

    def test_default_ids_are_positions(self):
        self.assertEqual(leave_one_out_sensitivity([1, 1, 1, 1, 10])["flagged_ids"], [4])

    def test_tiny_samples_are_handled(self):
        r = leave_one_out_sensitivity([5.0])
        self.assertEqual((r["flagged_ids"], len(r["loo_means"])), ([], 0))
        self.assertTrue(np.isnan(leave_one_out_sensitivity([])["full_mean"]))

    @unittest.expectedFailure
    def test_a_homogeneous_sample_flags_nobody(self):
        """KNOWN ISSUE: removing subject i shifts the mean by (x_i - mean)/(n-1), and the threshold
        (the SD of the leave-one-out means) is sd(x)/(n-1), so the rule "shift > threshold" is
        exactly "|x_i - mean| > 1 SD". It flags the subjects that are merely furthest from the mean
        in EVERY sample -- about 32% of subjects even in pure Gaussian noise, at every sample size
        (verified by simulation) -- so it cannot tell whether any subject is genuinely influential.
        Here a perfectly ordinary sample gets two subjects flagged."""
        self.assertEqual(leave_one_out_sensitivity([2.0, 2.1, 1.9, 2.05, 1.95])["flagged_ids"], [])

    @unittest.expectedFailure
    def test_pure_noise_does_not_flag_a_third_of_the_subjects(self):
        """KNOWN ISSUE (same cause): in 100 Gaussian subjects with no influential one, far fewer
        than 10% should be flagged as sensitive."""
        v = np.random.default_rng(0).normal(0, 1, 100)
        self.assertLess(len(leave_one_out_sensitivity(v)["flagged_ids"]) / 100, 0.10)


class TestPairedSimilarityAndPermutationTest(unittest.TestCase):
    def test_permutation_p_value_and_effect_size_by_hand(self):
        res = permutation_test_similarity(np.array([0.4, 0.6]), np.array([0.1, 0.2, 0.5, 0.6]))
        self.assertAlmostEqual(res["observed_mean_similarity"], 0.5)
        self.assertAlmostEqual(res["p_value"], (2 + 1) / (4 + 1))              # add-one smoothing, one-tailed
        null = np.array([0.1, 0.2, 0.5, 0.6])
        self.assertAlmostEqual(res["effect_size"], (0.5 - null.mean()) / null.std())

    def test_p_value_is_never_zero_and_a_constant_null_has_no_effect_size(self):
        res = permutation_test_similarity(np.array([0.9, 0.9]), np.array([0.1, 0.2, 0.3]))
        self.assertAlmostEqual(res["p_value"], 1 / 4)
        self.assertTrue(np.isnan(permutation_test_similarity(np.array([0.5]), np.array([0.3, 0.3]))["effect_size"]))

    def test_identical_fields_are_maximally_similar_and_clearly_beat_chance(self):
        emb = _unit_rows(40, 32, seed=5)
        out = compute_paired_similarity({"a": emb, "b": emb.copy()}, [("a", "b", "self")], n_null=200, rng_seed=0)["self"]
        assert_allclose(out["same_subject_similarity"], 1.0)
        self.assertLess(abs(out["null_mean"]), 0.1)                            # random pairing ~ 0 in 32 dims
        test = permutation_test_similarity(out["same_subject_similarity"], out["null_shuffle_means"])
        self.assertAlmostEqual(test["p_value"], 1 / 201)                       # smallest attainable with 200 shuffles
        self.assertGreater(test["effect_size"], 5)

    def test_unrelated_fields_are_indistinguishable_from_chance(self):
        emb = {"a": _unit_rows(60, 32, seed=6), "b": _unit_rows(60, 32, seed=7)}
        out = compute_paired_similarity(emb, [("a", "b", "unrelated")], n_null=500, rng_seed=1)["unrelated"]
        test = permutation_test_similarity(out["same_subject_similarity"], out["null_shuffle_means"])
        self.assertGreater(test["p_value"], 0.05)
        self.assertLess(abs(test["effect_size"]), 3)

    def test_output_shapes_and_seeding(self):
        emb = {"a": _unit_rows(10, 8, 1), "b": _unit_rows(10, 8, 2)}
        r1 = compute_paired_similarity(emb, [("a", "b", "p")], n_null=25, rng_seed=3)["p"]
        r2 = compute_paired_similarity(emb, [("a", "b", "p")], n_null=25, rng_seed=3)["p"]
        self.assertEqual(r1["same_subject_similarity"].shape, (10,))
        self.assertEqual(r1["null_values"].shape, (250,))
        self.assertEqual(r1["null_shuffle_means"].shape, (25,))
        assert_allclose(r1["null_shuffle_means"], r2["null_shuffle_means"])
        # each shuffle-mean really is the mean of that shuffle's 10 similarities
        assert_allclose(r1["null_shuffle_means"], r1["null_values"].reshape(25, 10).mean(axis=1))

    def test_delta_vector_is_to_minus_from_with_its_norm(self):
        emb = {"x": np.array([[1.0, 0.0], [0.0, 1.0]]), "y": np.array([[1.0, 1.0], [0.0, 3.0]])}
        vec, mag = compute_delta_vector(emb, "x", "y")
        assert_allclose(vec, [[0.0, 1.0], [0.0, 2.0]])
        assert_allclose(mag, [1.0, 2.0])


class TestWordcountChecks(unittest.TestCase):
    def _frame(self, n=200, seed=8):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({"wordcount_a": rng.uniform(5, 60, n), "wordcount_b": rng.uniform(5, 60, n)})
        df["sim_ab"] = 0.40 + 0.004 * df["wordcount_a"] + rng.normal(0, 0.02, n)   # depends on a, NOT on b
        return df

    def test_regression_matches_an_independent_least_squares_calculation(self):
        df = self._frame()
        out = wordcount_controlled_regression(df, "a", "b", "sim_ab")
        X = np.column_stack([np.ones(len(df)), df["wordcount_a"], df["wordcount_b"]])
        y = df["sim_ab"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - 3
        se = np.sqrt(np.diag(resid @ resid / dof * np.linalg.inv(X.T @ X)))
        pvals = 2 * stats.t.sf(np.abs(beta / se), dof)
        assert_allclose([out["coef_const"], out["coef_wordcount_a"], out["coef_wordcount_b"]], beta, rtol=1e-8)
        assert_allclose([out["pvalue_const"], out["pvalue_wordcount_a"], out["pvalue_wordcount_b"]], pvals, rtol=1e-6, atol=1e-12)
        self.assertAlmostEqual(out["r_squared"], 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())), places=10)

    def test_it_detects_the_dependence_that_was_built_in_and_not_the_one_that_was_not(self):
        out = wordcount_controlled_regression(self._frame(), "a", "b", "sim_ab")
        self.assertAlmostEqual(out["coef_wordcount_a"], 0.004, delta=0.0006)
        self.assertLess(out["pvalue_wordcount_a"], 1e-10)
        self.assertGreater(out["pvalue_wordcount_b"], 0.01)

    def test_too_few_subjects_gives_nans_instead_of_a_degenerate_fit(self):
        out = wordcount_controlled_regression(self._frame(n=2), "a", "b", "sim_ab")
        self.assertTrue(np.isnan(out["r_squared"]))
        self.assertIsNone(out["model"])

    def test_confound_check_matches_scipy_pearson(self):
        df = self._frame()
        got = wordcount_confound_check(df, [("a", "b", "ab")])["ab"]
        r, p = stats.pearsonr(df["wordcount_a"] + df["wordcount_b"], df["sim_ab"])
        self.assertAlmostEqual(got["r"], r, places=12)
        self.assertAlmostEqual(got["p_value"], p, places=12)

    def test_confound_check_is_nan_when_a_correlation_is_undefined(self):
        df = pd.DataFrame({"wordcount_a": [10, 10, 10], "wordcount_b": [5, 5, 5], "sim_ab": [0.1, 0.2, 0.3]})
        got = wordcount_confound_check(df, [("a", "b", "ab")])["ab"]
        self.assertTrue(np.isnan(got["r"]) and np.isnan(got["p_value"]))

    def test_low_quality_flags_use_a_strict_minimum_and_never_drop_rows(self):
        df = pd.DataFrame({"wordcount_a": [3, 5, 10], "wordcount_b": [20, 2, 30]})
        out = flag_low_quality(df, ["a", "b"], min_words=5)
        self.assertEqual(out["low_quality_a"].tolist(), [True, False, False])   # 5 words is NOT below 5
        self.assertEqual(out["low_quality_b"].tolist(), [False, True, False])
        self.assertEqual(out["any_low_quality"].tolist(), [True, True, False])
        self.assertEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main()
