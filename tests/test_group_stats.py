"""Group analysis statistics: the mixed models against the classical calculations they must reproduce
(paired t, one-sample t, repeated-measures ANOVA), known effects, edge cases, and the files written."""

import contextlib
import io
import os
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import scipy.stats as st

from PhysicsLibrary.analysis import group_stats as gs
from PhysicsLibrary.analysis import group_trials as gt
from PhysicsLibrary.analysis.group import GroupSpec
from PhysicsLibrary.analysis.group_report import CSV_NAMES, REPORT_FILENAME, report_text, write_group_results

MEASURES = ["auc", "peak", "mean", "latency", "decay"]


def cohort(seed=1, n_sub=8, markers=("A", "B"), n_trials=30, effects=None, sd_subj=1.0, sd_cell=0.5, sd_res=1.0,
           skip=None):
    """A trial table of `n_sub` subjects. AUC = effect + subject + subject x marker + noise; the other measures are
    transforms of it. `skip` = {(subject index, marker)} cells that are left out."""
    rng = np.random.default_rng(seed)
    effects = effects or {m: 0.0 for m in markers}
    skip = skip or set()
    rows = []
    for s in range(n_sub):
        u = rng.normal(0, sd_subj)
        for m in markers:
            c = rng.normal(0, sd_cell)
            if (s, m) in skip:
                continue
            n = n_trials[s] if isinstance(n_trials, (list, tuple)) else n_trials
            for t in range(n):
                y = effects[m] + u + c + rng.normal(0, sd_res)
                rows.append({"group": "G", "subject": f"S{s}", "recording": f"r{s}", "marker": m, "trial": t + 1,
                             "event_time": 10.0 * t, "valid": True, "reason": "", "auc": y, "peak": 2 * y, "mean": y / 3,
                             "latency": abs(y) + 1, "decay": np.nan if t % 5 == 0 else abs(y) + 2,
                             "baseline_mean": 0.0, "baseline_sd": 1.0, "n_samples": 100, "overlap": False})
    return pd.DataFrame(rows, columns=gt.TRIAL_COLUMNS)


def spec_for(markers, metrics=("auc",), **kw):
    return GroupSpec(group_name="G", markers=list(markers), metrics=list(metrics), **kw)


def cell_means(trials, measure="auc"):
    return trials.groupby(["subject", "marker"])[measure].mean().unstack()


def rm_anova_p(means):
    """Repeated-measures ANOVA p-value for the marker effect on a subjects x markers table of cell means."""
    n, k = means.shape
    grand = means.values.mean()
    ss_m = n * ((means.mean(axis=0) - grand) ** 2).sum()
    ss_s = k * ((means.mean(axis=1) - grand) ** 2).sum()
    ss_e = ((means.values - grand) ** 2).sum() - ss_m - ss_s
    f = (ss_m / (k - 1)) / (ss_e / ((k - 1) * (n - 1)))
    return f, st.f.sf(f, k - 1, (k - 1) * (n - 1))


def frame(results, name, **filters):
    df = getattr(results, name)
    for k, v in filters.items():
        df = df[df[k] == v]
    return df


class TestTwoMarkers(unittest.TestCase):
    def test_matches_the_paired_t_test_on_the_cell_means(self):
        trials = cohort(seed=3, n_sub=9, effects={"A": 0.0, "B": 0.7})
        res = gs.fit_group_models(trials, spec_for(["A", "B"]))
        means = cell_means(trials)
        paired = st.ttest_rel(means["A"], means["B"])
        for basis, tol in (("means", 1e-8), ("trials", 2e-4)):
            with self.subTest(basis=basis):
                row = frame(res, "pairwise", basis=basis).iloc[0]
                self.assertAlmostEqual(row["difference"], (means["A"] - means["B"]).mean(), places=6)
                self.assertAlmostEqual(row["t"], paired.statistic, delta=tol * 10)
                self.assertAlmostEqual(row["p_t"], paired.pvalue, delta=tol)
                self.assertEqual(row["df"], 8.0)
                self.assertEqual(row["n_subjects_with_both"], 9)
                om = frame(res, "omnibus", basis=basis).iloc[0]
                self.assertAlmostEqual(om["F"], paired.statistic ** 2, delta=tol * 100)        # with two markers F is t squared
                self.assertAlmostEqual(om["p_F"], paired.pvalue, delta=tol)
                self.assertEqual((om["df1"], om["df2"]), (1, 8))

    def test_estimated_means_are_the_marker_means_for_balanced_data(self):
        trials = cohort(seed=4, n_sub=7, effects={"A": 0.0, "B": 1.0})
        res = gs.fit_group_models(trials, spec_for(["A", "B"]))
        for basis in ("trials", "means"):
            emm = frame(res, "estimated_means", basis=basis).set_index("marker")
            for m in ("A", "B"):
                self.assertAlmostEqual(emm.loc[m, "mean"], trials[trials.marker == m]["auc"].mean(), places=5)
                self.assertEqual(emm.loc[m, "df"], 6.0)
                self.assertLess(emm.loc[m, "ci_low"], emm.loc[m, "mean"])
                self.assertGreater(emm.loc[m, "ci_high"], emm.loc[m, "mean"])

    def test_fixed_effects_are_the_reference_mean_and_the_difference_from_it(self):
        trials = cohort(seed=5, n_sub=8, effects={"A": 0.0, "B": 0.5})
        res = gs.fit_group_models(trials, spec_for(["A", "B"]))
        fe = frame(res, "fixed_effects", basis="trials")
        self.assertEqual(list(fe["term"]), ["mean of 'A'", "'B' minus 'A'"])
        self.assertAlmostEqual(fe.iloc[1]["estimate"], trials[trials.marker == "B"]["auc"].mean() - trials[trials.marker == "A"]["auc"].mean(), places=5)

    def test_variance_components_are_reported_with_their_shares(self):
        res = gs.fit_group_models(cohort(seed=6), spec_for(["A", "B"]))
        v = frame(res, "variance", basis="trials")
        self.assertEqual(list(v["component"]), ["subject", "subject x marker", "residual"])
        self.assertAlmostEqual(v["share_of_total"].sum(), 1.0, places=9)
        self.assertTrue((v["variance"] >= 0).all())
        self.assertEqual(list(frame(res, "variance", basis="means")["component"]), ["subject", "residual"])


class TestThreeMarkers(unittest.TestCase):
    def setUp(self):
        self.trials = cohort(seed=7, n_sub=8, markers=("A", "B", "C"), effects={"A": 0.0, "B": 0.8, "C": 1.6})
        self.res = gs.fit_group_models(self.trials, spec_for(["A", "B", "C"], correction="holm"))

    def test_omnibus_matches_repeated_measures_anova(self):
        f, p = rm_anova_p(cell_means(self.trials))
        for basis, tol in (("means", 1e-5), ("trials", 2e-3)):
            with self.subTest(basis=basis):
                om = frame(self.res, "omnibus", basis=basis).iloc[0]
                self.assertAlmostEqual(om["F"], f, delta=tol * max(1, f))
                self.assertAlmostEqual(om["p_F"], p, delta=tol)
                self.assertEqual((om["df1"], om["df2"]), (2, 14))

    def test_every_pair_is_compared_and_corrected(self):
        pw = frame(self.res, "pairwise", basis="trials")
        self.assertEqual([(a, b) for a, b in zip(pw["marker_a"], pw["marker_b"])], [("A", "B"), ("A", "C"), ("B", "C")])
        self.assertTrue((pw["p_adjusted"] >= pw["p_t"] - 1e-12).all())
        self.assertTrue((pw["p_adjusted"] <= 1).all())
        raw = pw["p_t"].to_numpy()
        order = np.argsort(raw)
        holm = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw[order])
        expected = np.empty(len(raw))
        expected[order] = np.minimum(holm, 1)
        np.testing.assert_allclose(pw["p_adjusted"].to_numpy(), expected, rtol=1e-9)
        self.assertEqual(list(pw["significant"]), list(pw["p_adjusted"] < 0.05))

    def test_pairwise_differences_are_differences_of_the_estimated_means(self):
        emm = frame(self.res, "estimated_means", basis="trials").set_index("marker")["mean"]
        for _, r in frame(self.res, "pairwise", basis="trials").iterrows():
            self.assertAlmostEqual(r["difference"], emm[r["marker_a"]] - emm[r["marker_b"]], places=9)

    def test_the_two_bases_agree_on_balanced_data(self):
        a = frame(self.res, "pairwise", basis="trials")["p_t"].to_numpy()
        b = frame(self.res, "pairwise", basis="means")["p_t"].to_numpy()
        np.testing.assert_allclose(a, b, atol=5e-3)

    def test_other_corrections(self):
        raw = frame(self.res, "pairwise", basis="trials")["p_t"].to_numpy()
        for method, check in (("bonferroni", lambda adj: np.allclose(adj, np.minimum(raw * 3, 1))),
                              ("fdr_bh", lambda adj: (adj >= raw - 1e-12).all() and (adj <= 1).all())):
            with self.subTest(method=method):
                res = gs.fit_group_models(self.trials, spec_for(["A", "B", "C"], correction=method))
                adj = frame(res, "pairwise", basis="trials")["p_adjusted"].to_numpy()
                self.assertTrue(check(adj))
        strict = gs.fit_group_models(self.trials, spec_for(["A", "B", "C"], alpha=1e-9))
        self.assertFalse(frame(strict, "pairwise")["significant"].any())

    def test_the_effect_of_marker_is_found_when_there_is_one_and_not_when_there_is_none(self):
        null = gs.fit_group_models(cohort(seed=11, n_sub=8, markers=("A", "B", "C")), spec_for(["A", "B", "C"]))
        self.assertGreater(frame(null, "omnibus", basis="trials").iloc[0]["p_F"], 0.05)
        self.assertLess(frame(self.res, "omnibus", basis="trials").iloc[0]["p_F"], 0.01)


class TestKnownEffects(unittest.TestCase):
    def test_a_large_cohort_recovers_the_true_differences(self):
        trials = cohort(seed=2, n_sub=40, markers=("A", "B", "C"), n_trials=40, effects={"A": 0.0, "B": 0.5, "C": 1.5})
        res = gs.fit_group_models(trials, spec_for(["A", "B", "C"], metrics=["auc", "peak", "mean"]))
        pw = frame(res, "pairwise", basis="trials", measure="auc").set_index(["marker_a", "marker_b"])
        self.assertAlmostEqual(pw.loc[("A", "B"), "difference"], -0.5, delta=0.2)
        self.assertAlmostEqual(pw.loc[("A", "C"), "difference"], -1.5, delta=0.2)
        self.assertAlmostEqual(pw.loc[("B", "C"), "difference"], -1.0, delta=0.2)
        peak = frame(res, "pairwise", basis="trials", measure="peak").set_index(["marker_a", "marker_b"])
        self.assertAlmostEqual(peak.loc[("A", "C"), "difference"], -3.0, delta=0.4)            # peak = 2 x auc in this table
        self.assertTrue(frame(res, "pairwise", measure="auc")["significant"].all())

    def test_estimates_and_p_values_do_not_depend_on_the_units(self):
        trials = cohort(seed=8, n_sub=8, markers=("A", "B", "C"), effects={"A": 0.0, "B": 0.4, "C": 0.9})
        base = gs.fit_group_models(trials, spec_for(["A", "B", "C"]))
        moved = trials.assign(auc=trials["auc"] * 1000.0 + 250.0)
        big = gs.fit_group_models(moved, spec_for(["A", "B", "C"]))
        for basis in ("trials", "means"):
            a, b = frame(base, "pairwise", basis=basis), frame(big, "pairwise", basis=basis)
            np.testing.assert_allclose(b["difference"].to_numpy(), 1000.0 * a["difference"].to_numpy(), rtol=1e-4)
            np.testing.assert_allclose(b["p_t"].to_numpy(), a["p_t"].to_numpy(), rtol=1e-3, atol=1e-6)
        tiny = gs.fit_group_models(trials.assign(auc=trials["auc"] * 1e-4), spec_for(["A", "B", "C"]))
        np.testing.assert_allclose(frame(tiny, "pairwise", basis="trials")["p_t"].to_numpy(),
                                   frame(base, "pairwise", basis="trials")["p_t"].to_numpy(), rtol=1e-3, atol=1e-6)


class TestOneMarker(unittest.TestCase):
    def setUp(self):
        self.trials = cohort(seed=9, n_sub=9, markers=("A",), effects={"A": 0.8})
        self.res = gs.fit_group_models(self.trials, spec_for(["A"], metrics=["auc", "mean", "peak", "latency", "decay"]))

    def test_auc_and_mean_are_tested_against_zero_like_a_one_sample_t_on_the_subject_means(self):
        for measure in ("auc", "mean"):
            with self.subTest(measure=measure):
                subj = self.trials.groupby("subject")[measure].mean()
                one = st.ttest_1samp(subj, 0.0)
                for basis, tol in (("means", 1e-8), ("trials", 1e-3)):
                    fe = frame(self.res, "fixed_effects", measure=measure, basis=basis).iloc[0]
                    self.assertAlmostEqual(fe["estimate"], subj.mean(), places=5)
                    self.assertAlmostEqual(fe["p_t"], one.pvalue, delta=tol)
                    self.assertEqual(fe["df"], 8.0)
                    half = st.t.ppf(0.975, 8) * subj.std(ddof=1) / 3.0
                    self.assertAlmostEqual(fe["ci_low"], subj.mean() - half, delta=tol)
                m = frame(self.res, "models", measure=measure, basis="trials").iloc[0]
                self.assertTrue(m["tested_against_zero"])

    def test_peak_latency_and_decay_have_estimates_and_intervals_but_no_p_value(self):
        for measure in ("peak", "latency", "decay"):
            with self.subTest(measure=measure):
                fe = frame(self.res, "fixed_effects", measure=measure, basis="trials").iloc[0]
                self.assertIn("estimate only", fe["term"])
                self.assertTrue(np.isnan(fe["p_t"]) and np.isnan(fe["p_z"]))
                self.assertTrue(np.isfinite(fe["ci_low"]) and np.isfinite(fe["ci_high"]))
                self.assertFalse(frame(self.res, "models", measure=measure, basis="trials").iloc[0]["tested_against_zero"])

    def test_there_is_no_omnibus_or_pairwise_table_for_one_marker(self):
        self.assertEqual(len(self.res.omnibus), 0)
        self.assertEqual(len(self.res.pairwise), 0)
        self.assertEqual(list(frame(self.res, "variance", measure="auc", basis="trials")["component"]), ["subject", "residual"])


class TestEdgeCases(unittest.TestCase):
    def test_a_subject_without_a_marker_still_contributes_to_the_others(self):
        trials = cohort(seed=12, n_sub=8, markers=("A", "B", "C"), skip={(0, "C"), (1, "C")}, effects={"A": 0.0, "B": 0.5, "C": 1.0})
        res = gs.fit_group_models(trials, spec_for(["A", "B", "C"]))
        pw = frame(res, "pairwise", basis="trials").set_index(["marker_a", "marker_b"])
        self.assertEqual(pw.loc[("A", "B"), "n_subjects_with_both"], 8)
        self.assertEqual(pw.loc[("A", "C"), "n_subjects_with_both"], 6)
        self.assertEqual(pw.loc[("A", "C"), "df"], 5.0)                                    # subjects with both, minus one
        om = frame(res, "omnibus", basis="trials").iloc[0]
        self.assertEqual(om["n_subjects_with_all_markers"], 6)
        self.assertEqual(om["df2"], 2 * 5)

    def test_unequal_trial_counts_fit(self):
        trials = cohort(seed=13, n_sub=6, n_trials=[10, 30, 25, 12, 40, 18], effects={"A": 0.0, "B": 0.6})
        res = gs.fit_group_models(trials, spec_for(["A", "B"]))
        self.assertEqual(frame(res, "models", basis="trials").iloc[0]["status"], "fitted")
        self.assertEqual(frame(res, "models", basis="trials").iloc[0]["n_observations"], 2 * (10 + 30 + 25 + 12 + 40 + 18))

    def test_a_marker_with_values_for_one_subject_only_is_left_out_with_a_note(self):
        trials = cohort(seed=14, n_sub=6, markers=("A", "B", "C"), skip={(s, "C") for s in range(1, 6)})
        res = gs.fit_group_models(trials, spec_for(["A", "B", "C"]))
        self.assertTrue(any("'C' has values for fewer than 2 subjects" in n["text"] for n in res.notes))
        self.assertEqual(set(frame(res, "pairwise")["marker_b"]), {"B"})
        self.assertEqual(frame(res, "models", basis="trials").iloc[0]["n_markers"], 2)

    def test_no_marker_with_two_subjects_is_not_fitted(self):
        trials = cohort(seed=15, n_sub=3, markers=("A",), skip={(1, "A"), (2, "A")})
        res = gs.fit_group_models(trials, spec_for(["A"]))
        self.assertTrue(all(res.models["status"] == "not fitted"))
        self.assertTrue(res.notes)

    def test_constant_data_cannot_be_modelled(self):
        trials = cohort(seed=16, n_sub=6).assign(auc=1.0)
        res = gs.fit_group_models(trials, spec_for(["A", "B"]))
        self.assertTrue(all(res.models["status"] == "not fitted"))
        self.assertTrue(any("identical" in n["text"] for n in res.notes))

    def test_identical_subjects_do_not_crash_the_fit(self):
        one = cohort(seed=17, n_sub=1)
        trials = pd.concat([one.assign(subject=f"S{i}", recording=f"r{i}") for i in range(4)], ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gs.fit_group_models(trials, spec_for(["A", "B"]))
        self.assertEqual(len(res.models), 2)                                               # both bases answered, however they came out

    def test_a_failed_trial_model_is_noted_and_the_means_check_still_runs(self):
        trials = cohort(seed=18, n_sub=7, effects={"A": 0.0, "B": 0.5})
        real = gs._fit_mixed
        with mock.patch.object(gs, "_fit_mixed", side_effect=lambda d, m, cell: (None, "boom") if cell else real(d, m, cell)):
            res = gs.fit_group_models(trials, spec_for(["A", "B"]))
        status = res.models.set_index("basis")["status"]
        self.assertEqual((status["trials"], status["means"]), ("not fitted", "fitted"))
        self.assertTrue(any("boom" in n["text"] and "subject-means analysis" in n["text"] for n in res.notes))
        self.assertEqual(len(frame(res, "pairwise", basis="trials")), 0)
        self.assertEqual(len(frame(res, "pairwise", basis="means")), 1)

    def test_invalid_trials_and_undefined_measures_are_left_out_and_noted(self):
        trials = cohort(seed=19, n_sub=6, effects={"A": 0.0, "B": 0.5}, n_trials=20)
        trials.loc[trials.index[:10], ["valid", "reason"]] = [False, "flat baseline (no spread to z-score by)"]
        trials.loc[trials.index[:10], "auc"] = np.nan
        res = gs.fit_group_models(trials, spec_for(["A", "B"], metrics=["auc", "decay"]))
        text = " ".join(n["text"] for n in res.notes)
        self.assertIn("10 of 240 trials could not be measured", text)
        self.assertIn("valid trials have no value", text)                                  # the decay time is empty for every 5th trial
        self.assertEqual(frame(res, "models", basis="trials", measure="auc").iloc[0]["n_observations"], 230)

    def test_the_diagnostics_line_up_with_their_rows_even_for_shuffled_data(self):
        trials = cohort(seed=20, n_sub=6, markers=("A", "B", "C")).sample(frac=1.0, random_state=1).reset_index(drop=True)
        res = gs.fit_group_models(trials, spec_for(["A", "B", "C"]))
        d = res.diagnostics
        np.testing.assert_allclose(d["observed"] - d["fitted"], d["resid"], atol=1e-9)
        spread = d.groupby(["subject", "marker"])["fitted"].agg(lambda v: v.max() - v.min())
        self.assertLess(spread.max(), 1e-9)                                                # one fitted value per subject x marker cell
        self.assertEqual(len(d), len(trials))
        merged = d.merge(trials[["subject", "marker", "auc"]].drop_duplicates(["subject", "marker"]), on=["subject", "marker"])
        self.assertEqual(len(merged.groupby(["subject", "marker"])), 18)

    def test_random_effects_are_one_number_per_subject(self):
        res = gs.fit_group_models(cohort(seed=21, n_sub=6), spec_for(["A", "B"]))
        re_ = frame(res, "random_effects", basis="trials")
        self.assertEqual(sorted(re_["subject"]), [f"S{i}" for i in range(6)])
        self.assertLess(abs(re_["effect"].mean()), 1.0)

    def test_progress_is_reported_per_measure_and_ends_at_a_hundred_percent(self):
        seen = []
        gs.fit_group_models(cohort(seed=22, n_sub=5), spec_for(["A", "B"], metrics=["auc", "peak", "mean"]),
                            progress=lambda f, m: seen.append((f, m)))
        self.assertEqual(seen[-1][0], 1.0)
        self.assertEqual([m for _, m in seen[:3]], ["Fitting models: AUC", "Fitting models: Peak amplitude", "Fitting models: Mean amplitude"])
        self.assertTrue(all(a[0] <= b[0] for a, b in zip(seen, seen[1:])))
        self.assertTrue(all(m.isascii() for _, m in seen))

    def test_it_never_leaks_library_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gs.fit_group_models(cohort(seed=23, n_sub=6, markers=("A", "B", "C")), spec_for(["A", "B", "C"]))
        self.assertEqual([str(w.message)[:60] for w in caught], [])


class TestWritingResults(unittest.TestCase):
    def setUp(self):
        self.trials = cohort(seed=24, n_sub=6, markers=("L1P¹", "Clap"), effects={"L1P¹": 0.0, "Clap": 0.6}, n_trials=12)
        self.spec = spec_for(["L1P¹", "Clap"], metrics=["auc", "latency"])
        self.spec.subjects = [{"subject": f"S{i}", "folder": f"f{i}"} for i in range(6)]
        grid = np.linspace(-10, 10, 51)
        traces = {f"S{i}": {m: {"n": 12, "mean": np.sin(grid) * (i + 1)} for m in self.spec.markers} for i in range(6)}
        self.res = gs.fit_group_models(self.trials, self.spec, traces=traces, trace_grid=grid, excluded={"Clap": 2, "L1P¹": 0})

    def test_every_table_is_written_as_csv_that_excel_and_python_can_read(self):
        with tempfile.TemporaryDirectory() as d:
            paths = write_group_results(self.res, os.path.join(d, "Sucrose"))
            names = {os.path.basename(p) for p in paths}
            self.assertTrue({CSV_NAMES["trials"], CSV_NAMES["pairwise"], CSV_NAMES["fixed_effects"], CSV_NAMES["omnibus"],
                             CSV_NAMES["variance"], CSV_NAMES["estimated_means"], CSV_NAMES["descriptives"],
                             CSV_NAMES["subject_means"], CSV_NAMES["diagnostics"], "mean_traces.csv", REPORT_FILENAME} <= names)
            back = pd.read_csv(os.path.join(d, "Sucrose", CSV_NAMES["trials"]), encoding="utf-8-sig")
            self.assertEqual(len(back), len(self.trials))
            self.assertIn("L1P¹", set(back["marker"]))
            self.assertTrue(open(os.path.join(d, "Sucrose", CSV_NAMES["trials"]), "rb").read(3) == b"\xef\xbb\xbf")
            tr = pd.read_csv(os.path.join(d, "Sucrose", "mean_traces.csv"), encoding="utf-8-sig")
            self.assertEqual(set(tr["subject"]), {f"S{i}" for i in range(6)})
            self.assertEqual(len(tr), 6 * 2 * 51)

    def test_the_report_states_the_design_the_results_and_the_methods(self):
        text = report_text(self.res)
        for needle in ("GROUP ANALYSIS: G", "Subjects (6)", "L1P¹", "AUC (dF/F x s)", "Latency to peak (s)", "Trial-level mixed model",
                       "Check on subject means", "Effect of marker: F(1, 5)", "Pairwise comparisons, p adjusted by Holm",
                       "METHODS", "statsmodels", "Events left out because their window"):
            self.assertIn(needle, text)
        self.assertNotIn("nan", text.lower().replace("nanoseconds", ""))

    def test_the_report_for_one_marker_says_what_is_and_is_not_tested(self):
        trials = cohort(seed=25, n_sub=6, markers=("A",), effects={"A": 0.5})
        spec = spec_for(["A"], metrics=["auc", "peak"])
        spec.subjects = [{"subject": f"S{i}", "folder": f"f{i}"} for i in range(6)]
        text = report_text(gs.fit_group_models(trials, spec))
        self.assertIn("against zero: estimate", text)
        self.assertIn("not tested against zero", text)

    def test_an_empty_analysis_still_writes_a_report(self):
        empty = gs.fit_group_models(cohort(seed=26, n_sub=3).iloc[0:0], spec_for(["A", "B"]))
        with tempfile.TemporaryDirectory() as d:
            write_group_results(empty, d)
            self.assertTrue(os.path.exists(os.path.join(d, REPORT_FILENAME)))


class TestEndToEndOnSimulatedPhotometry(unittest.TestCase):
    """Recordings simulated as traces, sliced and analysed: the pipeline must find the differences that were built in."""

    def run_cohort(self, n_sub=10, seed=0):
        rng = np.random.default_rng(seed)
        amp = {"big": 2.0, "small": 1.0}
        delay = {"big": 0.0, "small": 1.0}
        frames, traces = [], {}
        for s in range(n_sub):
            gain = rng.lognormal(0, 0.3)
            events = {m: list(np.sort(rng.uniform(30, 570, 24))) for m in amp}
            events = {m: [round(e, 2) for e in t] for m, t in events.items()}
            x = np.arange(60000) / 100.0
            y = 0.02 + 0.005 * rng.standard_normal(len(x))
            for m, times in events.items():
                for e in times:
                    s_ = x - (e + delay[m])
                    y = y + np.where(s_ >= 0, gain * amp[m] * np.exp(-np.clip(s_, 0, None) / 1.5), 0.0)
            spec = GroupSpec(group_name="G", markers=["big", "small"], pre=10, post=10, baseline=(-10, -6), response=(0, 10),
                             smooth_seconds=0.5, peak_direction="absolute", metrics=list(MEASURES))
            out = gt.extract_group_trials(x, y, events, spec, subject=f"S{s}", recording=f"r{s}")
            frames.append(out["trials"])
            traces[f"S{s}"] = out["traces"]
        spec.subjects = [{"subject": f"S{s}", "folder": f"f{s}"} for s in range(n_sub)]
        return gs.fit_group_models(pd.concat(frames, ignore_index=True), spec, traces=traces, trace_grid=out["grid"]), spec

    def test_the_built_in_differences_are_recovered(self):
        res, spec = self.run_cohort()
        pw = frame(res, "pairwise", basis="trials").set_index("measure")
        # the table's differences are 'big' minus 'small'
        self.assertGreater(pw.loc["peak", "difference"], 0.3)                   # 'big' has the larger response (2 vs 1, times the subject's gain)
        self.assertLess(pw.loc["peak", "difference"], 1.6)
        self.assertGreater(pw.loc["auc", "difference"], 0)
        self.assertLess(pw.loc["latency", "difference"], 0)                     # 'small' peaks about 1 s later (smoothing blurs it)
        self.assertGreater(pw.loc["latency", "difference"], -5)
        self.assertTrue(pw.loc["peak", "significant"] and pw.loc["auc", "significant"] and pw.loc["latency", "significant"])
        desc = res.descriptives.set_index(["measure", "marker"])
        self.assertAlmostEqual(desc.loc[("decay", "big"), "mean_of_subject_means"], 1.5 * np.log(2), delta=0.6)


if __name__ == "__main__":
    unittest.main()
