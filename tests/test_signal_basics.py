"""Small numeric building blocks: shared helpers, model functions, AUC, FFT,
marker intervals and splicing. Each test compares against a known answer."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import PhysicsLibrary as pl
from PhysicsLibrary import models
from PhysicsLibrary.analysis.auc import compute_auc_from_trace, compute_auc_window
from PhysicsLibrary.analysis.fft import compute_fft_slice, find_fft_peaks
from PhysicsLibrary.analysis.intervals import compute_marker_intervals
from PhysicsLibrary.analysis.shared import estimate_sample_rate, mean_channels, smooth_signal
from PhysicsLibrary.splice import splice_cut_out, splice_keep_inside


class TestSharedHelpers(unittest.TestCase):
    def test_estimate_sample_rate_regular_grid(self):
        t = np.arange(0, 10, 0.01)
        self.assertAlmostEqual(estimate_sample_rate(t), 100.0, places=6)

    def test_estimate_sample_rate_ignores_a_dropped_sample(self):
        # One missing sample doubles a single interval; the median ignores it.
        t = np.delete(np.arange(0, 10, 0.01), 500)
        self.assertAlmostEqual(estimate_sample_rate(t), 100.0, places=6)

    def test_mean_channels_averages_across_channels(self):
        arr = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
        assert_allclose(mean_channels(arr), [3.0, 4.0, 5.0])

    def test_mean_channels_leaves_1d_untouched(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert_array_equal(mean_channels(arr), arr)

    def test_smooth_signal_preserves_a_linear_ramp_in_the_interior(self):
        # A symmetric moving average of a straight line returns the line.
        fs = 100.0
        y = 0.3 * np.arange(1000)
        s = smooth_signal(y, fs, window_sec=0.5)   # 50 -> forced odd (51) samples
        assert_allclose(s[100:-100], y[100:-100], atol=1e-9)

    def test_smooth_signal_window_is_forced_odd(self):
        impulse = np.zeros(201)
        impulse[100] = 1.0
        s = smooth_signal(impulse, fs=100.0, window_sec=0.5)   # 50 samples requested
        self.assertEqual(int(np.count_nonzero(s)), 51)          # -> 51 (odd, centred)
        assert_allclose(s[s > 0], 1.0 / 51)

    def test_smooth_signal_removes_fast_noise_and_keeps_slow_signal(self):
        fs = 1000.0
        t = np.arange(0, 5, 1 / fs)
        slow = np.sin(2 * np.pi * 0.2 * t)
        fast = np.sin(2 * np.pi * 50 * t)
        s = smooth_signal(slow + fast, fs, window_sec=0.1)
        core = slice(500, -500)
        self.assertLess(np.abs(s[core] - slow[core]).max(), 0.06)

    def test_smooth_signal_edges_are_biased_low_by_zero_padding(self):
        # Documents current behaviour: np.convolve(mode="same") zero-pads, so the
        # first/last half-window are pulled toward 0. Interior samples are exact.
        s = smooth_signal(np.ones(1000), fs=100.0, window_sec=0.5)
        assert_allclose(s[100:-100], 1.0, atol=1e-12)
        self.assertLess(s[0], 0.6)


class TestModels(unittest.TestCase):
    def test_linear(self):
        assert_allclose(models.linear_model(np.array([0.0, 2.0]), 3.0, -1.0), [-1.0, 5.0])

    def test_single_exponential_limits(self):
        # y = a*exp(-b x) + c : equals a + c at x = 0 and decays to c.
        self.assertAlmostEqual(models.single_exponential_model(0.0, 4.0, 0.5, 1.5), 5.5)
        self.assertAlmostEqual(models.single_exponential_model(1e3, 4.0, 0.5, 1.5), 1.5)

    def test_exponential_rise_limits(self):
        # y = a*(1 - exp(-b x)) + c : equals c at x = 0 and rises to a + c.
        self.assertAlmostEqual(models.exponential_rise_model(0.0, 4.0, 0.5, 1.5), 1.5)
        self.assertAlmostEqual(models.exponential_rise_model(1e3, 4.0, 0.5, 1.5), 5.5)

    def test_double_exponential_limits(self):
        self.assertAlmostEqual(models.double_exponential_model(0.0, 2.0, 1.0, 3.0, 0.1, 7.0), 12.0)
        self.assertAlmostEqual(models.double_exponential_model(1e4, 2.0, 1.0, 3.0, 0.1, 7.0), 7.0)

    def test_gaussian_peak_and_fwhm(self):
        a, mu, sigma = 3.0, 2.0, 0.5
        self.assertAlmostEqual(models.gaussian_model(mu, a, mu, sigma), a)
        half_width = sigma * np.sqrt(2 * np.log(2))   # half of the FWHM
        self.assertAlmostEqual(models.gaussian_model(mu + half_width, a, mu, sigma), a / 2)

    def test_sinusoidal_period(self):
        f = 2.0
        x = np.linspace(0, 1, 11)
        assert_allclose(models.sinusoidal_model(x, 1.5, f, 0.3, 0.7),
                        models.sinusoidal_model(x + 1 / f, 1.5, f, 0.3, 0.7), atol=1e-9)

    def test_visibility_is_flat_at_zero_visibility(self):
        beta = np.linspace(0, 10, 7)
        assert_allclose(models.visibility_model(beta, 4.0, 0.0, 1.0, 2.0), 2.0)


class TestAUC(unittest.TestCase):
    def test_constant_positive_trace(self):
        x = np.linspace(0, 10, 101)
        r = compute_auc_from_trace(x, np.full_like(x, 2.0))
        self.assertAlmostEqual(r["total_auc"], 20.0)
        self.assertAlmostEqual(r["positive_auc"], 20.0)
        self.assertAlmostEqual(r["negative_auc"], 0.0)

    def test_negative_area_is_signed(self):
        x = np.linspace(0, 4, 401)
        r = compute_auc_from_trace(x, np.full_like(x, -3.0))
        self.assertAlmostEqual(r["total_auc"], -12.0)
        self.assertAlmostEqual(r["positive_auc"], 0.0)
        self.assertAlmostEqual(r["negative_auc"], -12.0)          # <= 0, not an absolute value

    def test_full_sine_period_splits_into_equal_and_opposite_lobes(self):
        # Integral of sin(2 pi x) over [0, 0.5] is 1/pi; the next half is -1/pi.
        x = np.linspace(0, 1, 20001)
        r = compute_auc_from_trace(x, np.sin(2 * np.pi * x))
        self.assertAlmostEqual(r["positive_auc"], 1 / np.pi, places=4)
        self.assertAlmostEqual(r["negative_auc"], -1 / np.pi, places=4)
        self.assertAlmostEqual(r["total_auc"], 0.0, places=4)

    def test_total_is_always_positive_plus_negative(self):
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 10, 500))
        y = rng.normal(0, 1, 500)
        r = compute_auc_from_trace(x, y)
        self.assertAlmostEqual(r["total_auc"], r["positive_auc"] + r["negative_auc"], places=10)

    def test_fewer_than_two_points_gives_zeros(self):
        self.assertEqual(compute_auc_from_trace(np.array([1.0]), np.array([5.0])),
                         {"total_auc": 0.0, "positive_auc": 0.0, "negative_auc": 0.0})

    def test_window_selects_the_requested_span(self):
        t = np.arange(0, 30, 0.01)
        sig = np.full_like(t, 3.0)
        r = compute_auc_window(t, sig, center_t=10.0, pre=2.0, post=5.0)
        self.assertAlmostEqual(r["seg_x"][0], 8.0, places=6)
        self.assertLess(r["seg_x"][-1], 15.0)
        # 3 units for ~7 seconds (the right edge is exclusive, so a hair under)
        self.assertAlmostEqual(r["total_auc"], 3.0 * 7.0, delta=0.05)


class TestFFT(unittest.TestCase):
    FS = 100.0

    def _tone(self, freq, seconds=60, amp=1.0):
        t = np.arange(0, seconds, 1 / self.FS)
        return t, amp * np.sin(2 * np.pi * freq * t)

    def test_single_tone_is_found_at_the_right_frequency(self):
        t, y = self._tone(1.5)
        freqs, power, _, _ = compute_fft_slice(t, y, center_t=30, fs=self.FS, pre=30, post=30)
        peaks = find_fft_peaks(freqs, power)
        self.assertGreaterEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0]["freq_hz"], 1.5, delta=0.03)      # bin width = fs/N ~ 0.017 Hz
        self.assertAlmostEqual(peaks[0]["bpm"], peaks[0]["freq_hz"] * 60)

    def test_two_tones_are_ranked_by_power(self):
        t, y1 = self._tone(1.0, amp=2.0)
        _, y2 = self._tone(3.0, amp=1.0)
        freqs, power, _, _ = compute_fft_slice(t, y1 + y2, 30, self.FS, pre=30, post=30)
        peaks = find_fft_peaks(freqs, power, n_peaks=2)
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(peaks[0]["freq_hz"], 1.0, delta=0.03)
        self.assertAlmostEqual(peaks[1]["freq_hz"], 3.0, delta=0.03)
        self.assertGreater(peaks[0]["power"], peaks[1]["power"])

    def test_offset_and_linear_drift_do_not_create_a_low_frequency_peak(self):
        t, y = self._tone(2.0)
        freqs, power, _, _ = compute_fft_slice(t, y + 50.0 + 0.5 * t, 30, self.FS, pre=30, post=30)
        peaks = find_fft_peaks(freqs, power)
        self.assertAlmostEqual(peaks[0]["freq_hz"], 2.0, delta=0.03)

    def test_symmetric_window_equals_pre_post(self):
        t, y = self._tone(1.5)
        a = compute_fft_slice(t, y, 30, self.FS, window=20)
        b = compute_fft_slice(t, y, 30, self.FS, pre=10, post=10)
        for u, v in zip(a, b):
            assert_allclose(u, v)

    def test_too_short_a_window_returns_empty(self):
        t, y = self._tone(1.5)
        # +/-15 ms at 100 Hz is 3 samples: fewer than the 4 an FFT needs here.
        freqs, power, seg_x, _ = compute_fft_slice(t, y, 30, self.FS, pre=0.015, post=0.015)
        self.assertEqual(len(seg_x), 3)
        self.assertEqual(len(freqs), 0)
        self.assertEqual(len(power), 0)
        self.assertEqual(find_fft_peaks(freqs, power), [])


class TestMarkerIntervals(unittest.TestCase):
    def test_intervals_per_store_and_globally(self):
        markers = [
            {"time": 6.0, "label": "PP1", "store": "PP1", "phase": "high"},
            {"time": 1.0, "label": "PP1", "store": "PP1", "phase": "high"},   # unsorted on purpose
            {"time": 2.5, "label": "PP1", "store": "PP1", "phase": "low"},
            {"time": 4.0, "label": "Clap"},                                   # no store: falls back to label
        ]
        rows = compute_marker_intervals(markers)
        self.assertEqual([r["time"] for r in rows], [1.0, 2.5, 4.0, 6.0])       # sorted
        self.assertEqual([r["dt_store"] for r in rows], [None, 1.5, None, 3.5])
        self.assertEqual([r["dt_global"] for r in rows], [None, 1.5, 1.5, 2.0])
        self.assertEqual(rows[2]["store"], "Clap")
        self.assertEqual(rows[2]["phase"], "")
        self.assertEqual(rows[1]["dt_store"], 1.5)            # a low marker's dt_store IS its on-duration


class TestSplice(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(0, 10, 0.5)                        # 20 samples, dt = 0.5
        self.raw = self.x * 10
        self.corr = self.x * 100
        self.markers = [{"time": 1.0, "id": "before"}, {"time": 3.0, "id": "inside"},
                        {"time": 7.0, "id": "after"}]

    def test_keep_inside_trims_signal_and_markers_with_absolute_times(self):
        out = splice_keep_inside(self.x, self.raw, self.corr, self.markers, [], 2.0, 5.0)
        assert_allclose(out["x"], np.arange(2.0, 5.5, 0.5))
        assert_allclose(out["raw"], out["x"] * 10)
        self.assertEqual(out["n_samples"], 7)
        self.assertEqual([m["id"] for m in out["markers"]], ["inside"])
        self.assertEqual(out["markers"][0]["time"], 3.0)      # NOT re-zeroed

    def test_keep_inside_returns_copies_not_views(self):
        out = splice_keep_inside(self.x, self.raw, self.corr, [], [], 2.0, 5.0)
        out["x"][0] = -999.0
        self.assertEqual(self.x[4], 2.0)

    def test_keep_inside_needs_at_least_two_samples(self):
        self.assertIsNone(splice_keep_inside(self.x, self.raw, self.corr, [], [], 2.0, 2.2))

    def test_keep_inside_slices_extra_channels_along_the_last_axis(self):
        two_d = np.vstack([self.x * 2, self.x * 3])
        out = splice_keep_inside(self.x, self.raw, self.corr, [], [], 2.0, 5.0,
                                 extra_channels={"a": self.x + 1, "b": two_d})
        assert_allclose(out["extra_channels"]["a"], out["x"] + 1)
        assert_allclose(out["extra_channels"]["b"], np.vstack([out["x"] * 2, out["x"] * 3]))

    def test_cut_out_stitches_a_gapless_timeline(self):
        out = splice_cut_out(self.x, self.raw, self.corr, self.markers, [], 2.0, 5.0)
        # samples at 2.0..5.0 inclusive are gone (7 of them)
        self.assertEqual(out["n_samples"], 13)
        assert_allclose(np.diff(out["x"]), 0.5)               # uniform sampling preserved: no gap at the cut
        assert_allclose(out["x"][:4], [0.0, 0.5, 1.0, 1.5])
        assert_allclose(out["x"][4:7], [2.0, 2.5, 3.0])       # what used to be 5.5, 6.0, 6.5
        # the signal values come from the original samples, just relabelled in time
        assert_allclose(out["raw"][4], 55.0)

    def test_cut_out_drops_inside_markers_and_shifts_later_ones(self):
        out = splice_cut_out(self.x, self.raw, self.corr, self.markers, [], 2.0, 5.0)
        by_id = {m["id"]: m["time"] for m in out["markers"]}
        self.assertNotIn("inside", by_id)
        self.assertEqual(by_id["before"], 1.0)
        self.assertEqual(by_id["after"], 3.5)                 # 7.0 - (5.5 - 2.0) cut duration

    def test_cut_out_can_start_at_the_very_beginning(self):
        out = splice_cut_out(self.x, self.raw, self.corr, [], [], -1.0, 3.0)
        self.assertIsNotNone(out)
        assert_allclose(out["x"][0], 0.0)                     # remaining data now begins at t = 0
        assert_allclose(np.diff(out["x"]), 0.5)

    def test_cut_out_can_run_to_the_very_end(self):
        out = splice_cut_out(self.x, self.raw, self.corr, [], [], 6.0, 99.0)
        assert_allclose(out["x"], np.arange(0, 6.0, 0.5))     # nothing after the cut, so nothing to shift

    def test_cut_out_refuses_to_leave_fewer_than_two_samples(self):
        self.assertIsNone(splice_cut_out(self.x, self.raw, self.corr, [], [], 0.0, 9.0))

    def test_cut_out_stitches_extra_channels_like_the_main_signal(self):
        two_d = np.vstack([self.raw, self.raw * 2])
        out = splice_cut_out(self.x, self.raw, self.corr, [], [], 2.0, 5.0, extra_channels={"m": two_d})
        assert_allclose(out["extra_channels"]["m"][0], out["raw"])
        assert_allclose(out["extra_channels"]["m"][1], out["raw"] * 2)


if __name__ == "__main__":
    unittest.main()
