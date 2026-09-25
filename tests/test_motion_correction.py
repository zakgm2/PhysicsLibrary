"""Motion correction (isosbestic regression), photobleaching correction, dF/F,
filtering, event debouncing and TDT epoc-marker extraction.

Two-channel data are simulated with a KNOWN dF/F so the pipeline's output can be
compared with the truth, not just inspected."""

import types
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose

from PhysicsLibrary.processing_TDT import (
    REGRESSION_METHODS, _robust_linear_fit, compute_dff, correct_bleaching,
    debounce_events, denoise_signal, get_event_markers,
)

FS = 20.0                      # transients are seconds wide, so 20 Hz keeps the tests fast
TRANSIENTS = ((100, 0.05), (200, 0.10), (300, 0.05), (400, 0.10), (500, 0.05))   # (centre s, true peak dF/F)


def _photometry(seed=1, seconds=600, artifact=False):
    """Two-channel recording with a KNOWN dF/F.

    465 nm = baseline(t) * (1 + true_dff) * motion(t) + noise
    415 nm = 0.5 * baseline(t) * motion(t) + noise
    where baseline(t) is exponential photobleaching and motion(t) is slow multiplicative
    motion shared by both channels (which is what an isosbestic control is for).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / FS)
    baseline = 60 + 90 * np.exp(-t / 300)
    true = np.zeros_like(t)
    for centre, amp in TRANSIENTS:
        true += amp * np.exp(-0.5 * ((t - centre) / 1.5) ** 2)
    motion = 1 + 0.02 * np.sin(t / 7.0) + 0.02 * rng.normal(0, 1, t.size).cumsum() / np.sqrt(t.size / 50)
    f465 = baseline * (1 + true) * motion + rng.normal(0, 0.15, t.size)
    f415 = 0.5 * baseline * motion + rng.normal(0, 0.15, t.size)
    if artifact:                                           # a 6 s fibre-twist-like burst in 465 only
        f465[(t > 250) & (t < 256)] += 25
    return t, f465, f415, true


def _peak(t, dff, centre, exclude=None):
    ok = np.ones_like(t, dtype=bool) if exclude is None else ~exclude
    return dff[(t > centre - 5) & (t < centre + 5)].max() - np.median(dff[ok])


class TestRegressionMethods(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.x = rng.uniform(10, 20, 6000)                     # "415 nm" values
        self.y = 2 * self.x + 5 + rng.normal(0, 0.05, self.x.size)   # "465 nm" = 2*415 + 5
        self.rng = rng

    def test_every_method_recovers_a_clean_line(self):
        for method in REGRESSION_METHODS:
            with self.subTest(method=method):
                a, b, frac = _robust_linear_fit(self.x, self.y, method)
                self.assertAlmostEqual(a, 2.0, delta=0.01)
                self.assertAlmostEqual(b, 5.0, delta=0.15)
                self.assertLessEqual(frac, 1.0)
        self.assertEqual(_robust_linear_fit(self.x, self.y, "ols")[2], 1.0)   # OLS never excludes anything

    def test_robust_methods_resist_random_artifacts_that_bias_ols(self):
        # 15% of samples get a large positive 465-only deviation at random positions.
        y = self.y.copy()
        hit = self.rng.choice(y.size, int(0.15 * y.size), replace=False)
        y[hit] += 30
        _, b_ols, _ = _robust_linear_fit(self.x, y, "ols")
        self.assertGreater(b_ols - 5.0, 3.0)                   # OLS intercept dragged ~4.5 up by the artifacts
        for method in ("ransac", "huber"):
            with self.subTest(method=method):
                a, b, _ = _robust_linear_fit(self.x, y, method)
                self.assertAlmostEqual(a, 2.0, delta=0.02)
                self.assertAlmostEqual(b, 5.0, delta=0.15)

    def test_ransac_excludes_about_the_artifact_fraction(self):
        y = self.y.copy()
        hit = self.rng.choice(y.size, int(0.15 * y.size), replace=False)
        y[hit] += 30
        _, _, frac = _robust_linear_fit(self.x, y, "ransac")
        self.assertAlmostEqual(frac, 0.85, delta=0.03)

    def test_the_fit_does_not_depend_on_the_units_of_the_data(self):
        """The same recording in other units must give the same line and the same inliers.
        (Huber used to stop early on data much larger than ~1, with a poor intercept and an
        inlier fraction of ~0.2%, because sklearn's optimiser is not scale-free.)"""
        base = {m: _robust_linear_fit(self.x, self.y, m) for m in REGRESSION_METHODS}
        for k in (1000.0, 0.001):
            for method in REGRESSION_METHODS:
                with self.subTest(method=method, k=k):
                    a, b, frac = _robust_linear_fit(k * self.x, k * self.y, method)
                    self.assertAlmostEqual(a, base[method][0], delta=1e-7)          # slope is dimensionless
                    self.assertAlmostEqual(b / k, base[method][1], delta=1e-7)      # intercept scales with the data
                    self.assertAlmostEqual(frac, base[method][2], delta=1e-7)       # so does who counts as an inlier

    def test_float32_input_gives_the_same_line_as_float64(self):
        """TDT streams are float32. np.polyfit on float32 data silently drops the slope term once
        n * float32-eps exceeds the ratio of the regressor's spread to its mean: here (415 nm
        channel = 25 +/- 1) from ~200,000 samples on, it returned slope ~0.95 and intercept ~24 for
        a true line of 1.9x + 0, with only a RankWarning. The OLS method (the default) and
        RANSAC's noise estimate both used it, so long recordings were motion-corrected against
        the wrong line."""
        rng = np.random.default_rng(31)
        n = 250_000
        x = (25 + rng.standard_normal(n)).astype(np.float32)
        y = (1.9 * x.astype(np.float64) + 0.3 * rng.standard_normal(n)).astype(np.float32)
        for method in REGRESSION_METHODS:
            with self.subTest(method=method):
                a, b, _ = _robust_linear_fit(x, y, method)
                self.assertAlmostEqual(a, 1.9, delta=0.02)
                self.assertAlmostEqual(b, 0.0, delta=0.5)

    def test_ransac_is_deterministic(self):
        first = _robust_linear_fit(self.x, self.y, "ransac")
        self.assertEqual(first, _robust_linear_fit(self.x, self.y, "ransac"))

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            _robust_linear_fit(self.x, self.y, "lasso")


class TestBleachingAndFiltering(unittest.TestCase):
    def test_bleaching_baseline_is_tracked_within_the_fitted_region_and_roughly_beyond(self):
        rng = np.random.default_rng(11)
        t = np.arange(0, 600, 1 / FS)
        truth = 40 + 100 * np.exp(-t / 150)
        _, trend = correct_bleaching(truth + rng.normal(0, 0.2, t.size), FS)
        rel_err = np.abs(trend / truth - 1)
        self.assertLess(rel_err[int(0 * FS)], 0.03)
        self.assertLess(rel_err[int(100 * FS)], 0.03)
        self.assertLess(rel_err.max(), 0.15)     # extrapolation error grows late in the recording (~10% at 10 min)

    def test_bleaching_correction_flattens_the_decay(self):
        rng = np.random.default_rng(11)
        t = np.arange(0, 600, 1 / FS)
        y = 40 + 100 * np.exp(-t / 150) + rng.normal(0, 0.2, t.size)
        corrected, _ = correct_bleaching(y, FS)
        self.assertLess(corrected.std(), 0.1 * y.std())

    def test_the_baseline_does_not_depend_on_the_units_of_the_signal(self):
        """The same trace in other units must give the same baseline (scaled). The double-
        exponential fit used to converge to a baseline up to ~4.5% different when the data
        were expressed x1000, because curve_fit's optimiser is not scale-free."""
        t, f465, _, _ = _photometry()
        _, trend = correct_bleaching(f465, FS)
        for k in (1000.0, 0.001):
            with self.subTest(k=k):
                assert_allclose(correct_bleaching(k * f465, FS)[1] / k, trend, rtol=1e-6)

    def test_too_few_samples_returns_the_input_and_a_zero_trend(self):
        y = np.linspace(10, 1, 150)                            # only 75 samples above the median (< 100)
        corrected, trend = correct_bleaching(y, FS)
        assert_allclose(corrected, y)
        assert_allclose(trend, 0.0)

    def test_denoise_keeps_slow_signal_and_removes_fast_noise_without_shifting_it(self):
        fs = 1000.0
        t = np.arange(0, 20, 1 / fs)
        slow = denoise_signal(np.sin(2 * np.pi * 0.1 * t), fs)
        fast = denoise_signal(np.sin(2 * np.pi * 40 * t), fs)
        self.assertGreater(np.abs(slow[2000:-2000]).max(), 0.99)
        self.assertLess(np.abs(fast[2000:-2000]).max(), 0.001)
        pulse = np.exp(-0.5 * ((t - 10) / 0.2) ** 2)
        self.assertLessEqual(abs(int(denoise_signal(pulse, fs).argmax()) - int(pulse.argmax())), 1)   # zero-phase


class TestDebounce(unittest.TestCase):
    def test_bounces_collapse_to_the_first_press_and_fast_real_presses_survive(self):
        kept = debounce_events([1.0, 0.0, 0.005, 0.3, 0.301, 0.45], min_isi=0.05)   # unsorted on purpose
        self.assertEqual(kept, [0.0, 0.3, 0.45, 1.0])       # 0.005 and 0.301 are bounces; 0.45 and 1.0 are real

    def test_events_exactly_min_isi_apart_are_both_kept(self):
        self.assertEqual(debounce_events([0.0, 0.05], min_isi=0.05), [0.0, 0.05])

    def test_empty(self):
        self.assertEqual(debounce_events([], 0.05), [])


class TestEventMarkers(unittest.TestCase):
    def _fake_block(self):
        ep = types.SimpleNamespace
        return types.SimpleNamespace(epocs={
            "Note": ep(onset=[1.0, 2.0], notes=[b"Clap", "Stop "]),
            "PP1_": ep(onset=[0.0, 5.0], offset=[6.0, np.inf]),
            "Tick": ep(onset=[1.0, 2.0, 3.0]),
            "Empty": ep(onset=[]),
        })

    def test_markers_are_extracted_per_store_with_phases(self):
        markers = get_event_markers(self._fake_block())
        notes = [m for m in markers if m["store"] == "Note"]
        self.assertEqual([(m["time"], m["label"]) for m in notes], [(1.0, "Clap"), (2.0, "Stop")])
        self.assertTrue(all("phase" not in m for m in notes))              # notes are instantaneous
        self.assertEqual(notes[0]["color"], "red")                          # experiment colour map for known notes
        pp1 = [m for m in markers if m["store"] == "PP1"]                   # trailing '_' stripped
        self.assertEqual([(m["time"], m["phase"]) for m in pp1], [(5.0, "high"), (6.0, "low")])

    def test_heartbeat_synthetic_onset_and_infinite_offset_are_dropped(self):
        markers = get_event_markers(self._fake_block())
        self.assertFalse(any(m["store"].lower() == "tick" for m in markers))
        self.assertFalse(any(m["time"] == 0.0 for m in markers))            # synthetic t=0 onset of a level store
        self.assertFalse(any(not np.isfinite(m["time"]) for m in markers))  # offset = inf (still active at the end)
        self.assertFalse(any(m["store"] == "Empty" for m in markers))

    def test_a_block_without_epocs_gives_no_markers(self):
        self.assertEqual(get_event_markers(types.SimpleNamespace()), [])


class TestComputeDff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.t, cls.f465, cls.f415, cls.true = _photometry()

    def test_contract_for_a_single_channel_recording(self):
        out = compute_dff(self.f465, None, FS)
        self.assertEqual(set(out), {"raw", "corr", "dff", "f0", "motion_correction_inlier_fraction"})
        self.assertIsNone(out["motion_correction_inlier_fraction"])         # nothing to regress against
        assert_allclose(out["corr"], out["dff"])
        self.assertEqual(len(out["dff"]), len(self.f465))

    def test_two_channel_inlier_fraction_is_reported(self):
        self.assertEqual(compute_dff(self.f465, self.f415, FS, "ols")["motion_correction_inlier_fraction"], 1.0)
        t, f465, f415, _ = _photometry(artifact=True)
        frac = compute_dff(f465, f415, FS, "ransac")["motion_correction_inlier_fraction"]
        self.assertTrue(0.5 < frac < 1.0)

    def test_transient_timing_and_shape_are_recovered_by_every_method(self):
        for method in REGRESSION_METHODS:
            with self.subTest(method=method):
                dff = compute_dff(self.f465, self.f415, FS, method)["dff"]
                self.assertGreater(np.corrcoef(self.true, dff)[0, 1], 0.9)

    def test_dff_amplitudes_match_the_true_dff(self):
        """The result is a real delta-F/F: true peaks of 5% and 10% come out as ~5% and ~10% with
        the baseline at ~0, for every regression method. (Regression guard: compute_dff used to
        divide the motion-corrected residual by the bleaching trend of THAT SAME near-zero-
        centred residual; on this data it reported +916%/+1,594% for RANSAC, +941%/+1,650% for
        Huber and +246,000%/+399,000% for OLS, with the baseline at about -1 (OLS: -66).) The
        tolerance covers the ~6% error of the double-exponential baseline, which is
        extrapolated over the second half of the trace."""
        for method in REGRESSION_METHODS:
            with self.subTest(method=method):
                dff = compute_dff(self.f465, self.f415, FS, method)["dff"]
                self.assertLess(abs(np.median(dff)), 0.01)                  # baseline sits at ~0
                self.assertAlmostEqual(_peak(self.t, dff, 100), 0.05, delta=0.005)
                self.assertAlmostEqual(_peak(self.t, dff, 200), 0.10, delta=0.01)

    def test_dff_does_not_depend_on_the_units_of_the_recording(self):
        """delta-F/F is a ratio, so multiplying both channels by a constant must not change it."""
        for method in REGRESSION_METHODS:
            with self.subTest(method=method):
                a = compute_dff(self.f465, self.f415, FS, method)["dff"]
                for k in (1000.0, 0.001):
                    b = compute_dff(k * self.f465, k * self.f415, FS, method)["dff"]
                    assert_allclose(a, b, rtol=0, atol=1e-8)

    def test_regression_methods_agree_on_clean_data(self):
        """On artifact-free data all three regressions fit almost the same line, so their
        delta-F/F must agree closely (before the fix their scale differed by orders of magnitude
        -- ~800x between OLS and RANSAC on a real recording -- so the choice of regression
        method changed the numbers far more than it should)."""
        d = {m: compute_dff(self.f465, self.f415, FS, m)["dff"] for m in REGRESSION_METHODS}
        for other in ("ransac", "huber"):
            self.assertGreater(np.corrcoef(d["ols"], d[other])[0, 1], 0.99)
            assert_allclose(d["ols"].std(), d[other].std(), rtol=0.1)

    def test_a_long_float32_recording_gives_the_same_dff_as_float64(self):
        """The same recording handed over as float32 (which is how TDT delivers it) must give the
        same dF/F as in float64. 300,000 samples is past the point where np.polyfit's float32
        fit used to collapse to the wrong line (see the regression tests above)."""
        rng = np.random.default_rng(5)
        fs = 1000.0
        t = np.arange(0, 300, 1 / fs)
        baseline = 25 + 3 * np.exp(-t / 200)                      # the 415 channel: 25 +/- ~1
        motion = 1 + 0.02 * np.sin(t / 5.0)
        true = 0.05 * np.exp(-0.5 * ((t - 150) / 1.5) ** 2)
        f415 = baseline * motion + rng.normal(0, 0.05, t.size)
        f465 = 1.9 * baseline * (1 + true) * motion + rng.normal(0, 0.05, t.size)
        dff64 = compute_dff(f465, f415, fs, "ols")["dff"]
        dff32 = compute_dff(f465.astype(np.float32), f415.astype(np.float32), fs, "ols")["dff"]
        assert_allclose(dff32, dff64, atol=1e-3)
        self.assertAlmostEqual(_peak(t, dff32, 150), 0.05, delta=0.01)

    def test_f0_is_the_baseline_fluorescence_and_dff_is_the_residual_over_it(self):
        baseline = 60 + 90 * np.exp(-self.t / 300)
        for method in REGRESSION_METHODS:
            with self.subTest(method=method):
                out = compute_dff(self.f465, self.f415, FS, method)
                self.assertLess(np.abs(out["f0"] / baseline - 1).max(), 0.10)      # a fluorescence level, not ~0
                assert_allclose(out["dff"], denoise_signal(out["raw"] / out["f0"], FS, cutoff=5))

    def test_single_channel_dff_is_the_signal_relative_to_its_own_baseline(self):
        out = compute_dff(self.f465, None, FS)
        assert_allclose(out["raw"], self.f465)
        assert_allclose(out["dff"], denoise_signal((self.f465 - out["f0"]) / out["f0"], FS, cutoff=5))

    def test_a_partial_dropout_does_not_blow_up_the_dff(self):
        """Both channels sag towards zero for a moment (a loose connector, a cable knock, the LEDs
        still switching on at the start of a real recording -- there the 415 comes up about half
        a second before the 465). Normalising by the fitted isosbestic (a*F415 + b), the textbook
        formula, divides by ~0 there: it gives +170% to +430% on this data and absurd values
        (SD ~1e5) on the real recording this was checked on, where the fitted line dips below
        zero. Normalising by the smooth bleaching baseline does not."""
        f465, f415 = self.f465.copy(), self.f415.copy()
        gap = (self.t > 300) & (self.t < 302)
        f465[gap], f415[gap] = 0.5, 0.0
        for method in REGRESSION_METHODS:
            with self.subTest(method=method):
                dff = compute_dff(f465, f415, FS, method)["dff"]
                self.assertLess(np.abs(dff).max(), 1.5)

    @unittest.expectedFailure
    def test_single_channel_dff_tracks_the_transients(self):
        """KNOWN LIMITATION: without an isosbestic channel the baseline comes only from
        correct_bleaching, which fits the double exponential to the samples ABOVE the median (for a
        decaying trace: the early half) and extrapolates -- up to ~10% error late in a 10-minute
        recording, which swamps 5-10% transients. Here the recovered trace correlates only ~0.3 with
        the true delta-F/F."""
        rng = np.random.default_rng(3)
        baseline = 60 + 90 * np.exp(-self.t / 300)
        dff = compute_dff(baseline * (1 + self.true) + rng.normal(0, 0.15, self.t.size), None, FS)["dff"]
        self.assertGreater(np.corrcoef(self.true, dff)[0, 1], 0.7)


if __name__ == "__main__":
    unittest.main()
