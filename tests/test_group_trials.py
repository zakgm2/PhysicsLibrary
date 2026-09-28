"""Group analysis trials: the five measures against analytic answers, and slicing a recording into baseline-corrected trials."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from PhysicsLibrary.analysis import group_trials as gt
from PhysicsLibrary.analysis.group import GroupSpec


def spec(**kw):
    base = dict(group_name="G", markers=["A"], pre=10.0, post=10.0, baseline=(-10.0, -6.0), response=(0.0, 10.0),
                signal="dff", smooth_seconds=0.0, peak_direction="absolute", decay_fraction=0.5)
    base.update(kw)
    return GroupSpec(**base)


class TestDecayTime(unittest.TestCase):
    def test_interpolates_between_the_two_samples_around_the_crossing(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 4.0, 3.0, 1.0])                       # peak 4 at t=1; half = 2 lies between t=2 (3) and t=3 (1)
        self.assertAlmostEqual(gt.decay_time(t, y, 1, 0.5), 1.5)

    def test_negative_peak_falls_back_when_it_climbs_to_the_level(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, -4.0, -3.0, -1.0])                    # dip of -4: half-way back is -2, between t=2 and t=3
        self.assertAlmostEqual(gt.decay_time(t, y, 1, 0.5), 1.5)

    def test_never_falling_far_enough_is_nan(self):
        t = np.arange(5.0)
        self.assertTrue(np.isnan(gt.decay_time(t, np.array([0, 4, 4, 4, 3.0]), 1, 0.5)))

    def test_a_peak_at_the_last_sample_and_a_zero_peak_are_nan(self):
        t = np.arange(4.0)
        self.assertTrue(np.isnan(gt.decay_time(t, np.array([0, 1, 2, 3.0]), 3, 0.5)))
        self.assertTrue(np.isnan(gt.decay_time(t, np.zeros(4), 1, 0.5)))

    def test_exponential_decay_gives_tau_times_the_log(self):
        t = np.arange(0, 20, 0.01)
        y = 2.0 * np.exp(-t / 2.0)
        self.assertAlmostEqual(gt.decay_time(t, y, 0, 0.5), 2.0 * np.log(2), places=3)      # half-decay time
        self.assertAlmostEqual(gt.decay_time(t, y, 0, 0.25), 2.0 * np.log(4), places=3)

    def test_gaussian_bump_falls_to_half_at_1_1774_sigma(self):
        t = np.arange(-10, 10, 0.005)
        y = np.exp(-0.5 * (t / 1.0) ** 2)
        k = int(np.argmax(y))
        self.assertAlmostEqual(gt.decay_time(t, y, k, 0.5), 1.1774, places=3)


class TestMeasuresForTrial(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(-1000, 1001) / 100.0                                  # an exact 100 Hz grid, t = 0 included
        self.y = np.where(self.t >= 0, 2.0 * np.exp(-self.t / 2.0), 0.0)        # a step-and-decay response, tau = 2 s

    def test_auc_peak_mean_latency_and_half_decay_match_the_analytic_values(self):
        m = gt.measures_for_trial(self.t, self.y, spec())
        self.assertAlmostEqual(m["auc"], 2.0 * 2.0 * (1 - np.exp(-5)), places=3)
        self.assertAlmostEqual(m["peak"], 2.0, places=6)
        self.assertAlmostEqual(m["latency"], 0.0, places=6)
        self.assertAlmostEqual(m["mean"], m["auc"] / 10.0, delta=2e-3)                 # the mean counts both end samples, the trapezoid halves them
        self.assertAlmostEqual(m["decay"], 2.0 * np.log(2), places=2)

    def test_the_response_window_limits_what_is_measured_but_not_where_the_decay_is_looked_for(self):
        m = gt.measures_for_trial(self.t, self.y, spec(response=(0.0, 2.0)))
        self.assertAlmostEqual(m["auc"], 4.0 * (1 - np.exp(-1)), places=3)              # only the first 2 s are integrated
        self.assertAlmostEqual(m["decay"], 2.0 * np.log(2), places=2)                   # yet the half-decay (1.39 s) is found

    def test_peak_direction_picks_the_positive_negative_or_largest_deflection(self):
        y = np.zeros_like(self.t)
        y[(self.t > 1.9) & (self.t < 2.1)] = 1.0
        y[(self.t > 4.9) & (self.t < 5.1)] = -3.0
        pos = gt.measures_for_trial(self.t, y, spec(peak_direction="positive"))
        neg = gt.measures_for_trial(self.t, y, spec(peak_direction="negative"))
        both = gt.measures_for_trial(self.t, y, spec(peak_direction="absolute"))
        self.assertEqual((pos["peak"], round(pos["latency"], 2)), (1.0, 1.91))
        self.assertEqual((neg["peak"], round(neg["latency"], 2)), (-3.0, 4.91))
        self.assertEqual((both["peak"], round(both["latency"], 2)), (-3.0, 4.91))          # signed, as Event PETH reports it

    def test_a_dip_decays_back_up(self):
        m = gt.measures_for_trial(self.t, -self.y, spec(peak_direction="negative"))
        self.assertAlmostEqual(m["peak"], -2.0, places=6)
        self.assertAlmostEqual(m["decay"], 2.0 * np.log(2), places=2)

    def test_a_positive_peak_that_is_below_zero_has_no_decay(self):
        y = -np.abs(self.y) - 0.1                                                          # never above zero in the window
        m = gt.measures_for_trial(self.t, y, spec(peak_direction="positive"))
        self.assertLess(m["peak"], 0)
        self.assertTrue(np.isnan(m["decay"]))                                             # the largest value is the last sample: nothing after it

    def test_fewer_than_two_samples_in_the_response_window_gives_none(self):
        self.assertIsNone(gt.measures_for_trial(self.t, self.y, spec(response=(0.0, 0.004))))


def recording(events, amp, tau=2.0, delay=0.0, offset=0.2, noise=0.0, T=300.0, fs=100.0, seed=0, t0=0.0):
    """A dF/F trace with a decaying response after every event: (x, y)."""
    rng = np.random.default_rng(seed)
    x = t0 + np.arange(int(T * fs)) / fs
    y = offset + noise * rng.standard_normal(len(x))
    for marker, times in events.items():
        for e in times:
            s = x - (e + delay)
            y = y + np.where(s >= 0, amp[marker] * np.exp(-np.clip(s, 0, None) / tau), 0.0)
    return x, y


class TestExtractGroupTrials(unittest.TestCase):
    EVENTS = {"A": [50.0, 100.0, 150.0], "B": [75.0, 125.0]}
    AMP = {"A": 1.0, "B": 2.0}

    def run_it(self, sp=None, events=None, **kw):
        events = events or self.EVENTS
        x, y = recording(events, self.AMP, **kw)
        return gt.extract_group_trials(x, y, events, sp or spec(markers=["A", "B"]), subject="S1", recording="r1")

    def test_one_row_per_usable_trial_in_marker_then_time_order(self):
        out = self.run_it()
        tr = out["trials"]
        self.assertEqual(list(tr.columns), gt.TRIAL_COLUMNS)
        self.assertEqual(list(tr["marker"]), ["A", "A", "A", "B", "B"])
        self.assertEqual(list(tr["trial"]), [1, 2, 3, 1, 2])
        self.assertEqual(list(tr["event_time"]), [50.0, 100.0, 150.0, 75.0, 125.0])
        self.assertTrue(tr["valid"].all())
        self.assertEqual(set(tr["subject"]), {"S1"})
        self.assertEqual(set(tr["group"]), {"G"})
        self.assertEqual(out["excluded"], {"A": 0, "B": 0})

    def test_the_measures_are_those_of_the_response_whatever_the_baseline_offset(self):
        for offset in (0.2, 5.0, -3.0):
            with self.subTest(offset=offset):
                tr = self.run_it(offset=offset)["trials"]
                a, b = tr[tr.marker == "A"].iloc[0], tr[tr.marker == "B"].iloc[0]
                self.assertAlmostEqual(a["peak"], 1.0, places=3)
                self.assertAlmostEqual(b["peak"], 2.0, places=3)
                self.assertAlmostEqual(b["auc"], 2.0 * 2.0 * (1 - np.exp(-5)), places=2)
                self.assertAlmostEqual(b["decay"], 2.0 * np.log(2), places=2)
                self.assertAlmostEqual(a["baseline_mean"], offset, places=3)

    def test_latency_follows_a_delayed_response(self):
        tr = self.run_it(delay=1.5)["trials"]
        self.assertAlmostEqual(tr["latency"].iloc[0], 1.5, places=2)

    def test_events_without_a_full_window_are_left_out_and_counted(self):
        events = {"A": [3.0, 50.0, 296.0], "B": []}
        out = self.run_it(events=events)
        self.assertEqual(list(out["trials"]["event_time"]), [50.0])
        self.assertEqual(out["excluded"], {"A": 2, "B": 0})
        self.assertEqual(out["traces"]["B"]["n"], 0)
        self.assertTrue(np.isnan(out["traces"]["B"]["mean"]).all())

    def test_a_recording_that_does_not_start_at_zero(self):
        events = {"A": [150.0, 200.0]}
        x, y = recording(events, self.AMP, t0=100.0)                                        # x runs 100..400, like a Keep Inside splice
        tr = gt.extract_group_trials(x, y, events, spec(), subject="S")["trials"]
        self.assertEqual(len(tr), 2)
        self.assertAlmostEqual(tr["peak"].iloc[0], 1.0, places=3)
        edge = gt.extract_group_trials(x, y, {"A": [105.0]}, spec(), subject="S")
        self.assertEqual((len(edge["trials"]), edge["excluded"]["A"]), (0, 1))

    def test_overlap_marks_trials_whose_windows_share_samples(self):
        events = {"A": [40.0, 59.9, 100.0, 120.0]}                                          # window is 20 s: 19.9 apart overlaps, 20.0 does not
        tr = self.run_it(events=events, sp=spec(markers=["A"]))["trials"]
        self.assertEqual(list(tr["overlap"]), [True, True, False, False])

    def test_unsorted_events_are_sorted(self):
        tr = self.run_it(events={"A": [150.0, 50.0, 100.0]}, sp=spec(markers=["A"]))["trials"]
        self.assertEqual(list(tr["event_time"]), [50.0, 100.0, 150.0])

    def test_mean_trace_is_the_average_of_the_valid_trials_on_a_common_axis(self):
        out = self.run_it()
        grid, tr = out["grid"], out["traces"]["B"]
        self.assertAlmostEqual(grid[0], -10.0)
        self.assertAlmostEqual(grid[-1], 10.0, places=6)
        self.assertEqual(tr["n"], 2)
        i0 = int(np.argmin(np.abs(grid - 0.0)))
        self.assertAlmostEqual(tr["mean"][i0], 2.0, delta=0.05)
        self.assertAlmostEqual(tr["mean"][0], 0.0, delta=2e-3)                              # baseline-corrected: zero before the event
        i5 = int(np.argmin(np.abs(grid - 2.0)))
        self.assertAlmostEqual(tr["mean"][i5], 2.0 * np.exp(-1), delta=0.02)

    def test_zscore_divides_by_the_baseline_spread_and_a_flat_baseline_cannot_be_scored(self):
        z = self.run_it(sp=spec(markers=["B"], signal="zscore"), noise=0.02)["trials"].iloc[0]
        d = self.run_it(sp=spec(markers=["B"], signal="dff"), noise=0.02)["trials"].iloc[0]
        self.assertTrue(z["valid"])
        self.assertAlmostEqual(z["peak"], d["peak"] / z["baseline_sd"], places=6)
        flat = self.run_it(sp=spec(markers=["B"], signal="zscore"), events={"B": [125.0]}, noise=0.0)["trials"]      # no earlier event: a truly flat baseline
        self.assertFalse(flat["valid"].any())
        self.assertTrue(all("flat baseline" in r for r in flat["reason"]))
        self.assertTrue(flat["auc"].isna().all())
        ok = self.run_it(sp=spec(markers=["B"], signal="dff"), events={"B": [125.0]}, noise=0.0)["trials"]           # the same recording is fine as dF/F
        self.assertTrue(ok["valid"].all())

    def test_a_spike_inside_the_baseline_does_not_move_the_baseline_mean(self):
        events = {"A": [100.0]}
        x, y = recording(events, self.AMP, offset=0.0, noise=0.01)
        y[np.argmin(np.abs(x - 92.0))] = 50.0                                               # a single motion spike at -8 s
        tr = gt.extract_group_trials(x, y, events, spec(), subject="S")["trials"].iloc[0]
        self.assertLess(abs(tr["baseline_mean"]), 0.05)
        self.assertAlmostEqual(tr["peak"], 1.0, delta=0.1)

    def test_non_finite_samples_make_the_trial_invalid_with_a_reason(self):
        events = {"A": [50.0, 100.0]}
        x, y = recording(events, self.AMP)
        y[np.argmin(np.abs(x - 103.0))] = np.nan
        tr = gt.extract_group_trials(x, y, events, spec(), subject="S")["trials"]
        self.assertEqual(list(tr["valid"]), [True, False])
        self.assertIn("non-finite", tr["reason"].iloc[1])

    def test_a_baseline_window_with_no_samples_and_a_one_sample_response_are_reported(self):
        events = {"A": [100.0]}
        x, y = recording(events, self.AMP, fs=2.0)                                          # 2 Hz: samples every 0.5 s
        tr = gt.extract_group_trials(x, y, events, spec(baseline=(-9.8, -9.6)), subject="S")["trials"]
        self.assertIn("no samples in the baseline window", tr["reason"].iloc[0])
        tr = gt.extract_group_trials(x, y, events, spec(response=(0.0, 0.2)), subject="S")["trials"]
        self.assertIn("fewer than 2 samples", tr["reason"].iloc[0])

    def test_smoothing_lowers_a_sharp_peak_and_matches_the_event_peth_smoother(self):
        events = {"A": [100.0]}
        x = np.arange(0, 200, 0.01)
        y = np.zeros_like(x)
        y[np.argmin(np.abs(x - 100.0))] = 1.0                                               # one-sample spike
        raw = gt.extract_group_trials(x, y, events, spec(smooth_seconds=0.0), subject="S")["trials"].iloc[0]
        smooth = gt.extract_group_trials(x, y, events, spec(smooth_seconds=0.5), subject="S")["trials"].iloc[0]
        self.assertAlmostEqual(raw["peak"], 1.0)
        self.assertAlmostEqual(smooth["peak"], 1.0 / 51, places=6)                          # a 0.5 s window at 100 Hz is 51 samples

    def test_markers_not_in_the_spec_are_ignored_and_unknown_ones_give_no_rows(self):
        out = self.run_it(sp=spec(markers=["B", "Z"]))
        self.assertEqual(set(out["trials"]["marker"]), {"B"})
        self.assertEqual(out["excluded"]["Z"], 0)
        self.assertEqual(out["traces"]["Z"]["n"], 0)

    def test_nothing_to_extract_gives_an_empty_table_with_the_columns(self):
        x, y = recording({}, {})
        out = gt.extract_group_trials(x, y, {}, spec(), subject="S")
        self.assertEqual(list(out["trials"].columns), gt.TRIAL_COLUMNS)
        self.assertEqual(len(out["trials"]), 0)


if __name__ == "__main__":
    unittest.main()
