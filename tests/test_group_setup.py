"""Group analysis, the parts that come before any signal is loaded: replaying saved splices on marker
times, listing a TDT block's markers without processing it, the settings object, and the design checks
(which markers the recordings share, how many usable trials each subject has, overlap)."""

import datetime
import json
import types
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import tdt

import PhysicsLibrary as pl
from PhysicsLibrary import processing_TDT as P
from PhysicsLibrary.analysis import group
from PhysicsLibrary.splice import MODE_CUT_OUT, MODE_KEEP_INSIDE, replay_splices, splice_cut_out, splice_keep_inside


def _marker(t, store="L1P", phase="high"):
    return {"time": float(t), "label": store, "color": "black", "store": store, "phase": phase}


class TestReplaySplices(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(10000) / 100.0                    # 100 s at 100 Hz, an exact grid (1000/100.0 == 10.0)
        self.markers = [_marker(t) for t in (5.0, 20.0, 55.5, 90.0)]

    def test_no_splices_changes_nothing(self):
        out = replay_splices(self.x, [], detected_markers=self.markers)
        assert_array_equal(out["x"], self.x)
        self.assertEqual(out["detected_markers"], self.markers)
        self.assertEqual(out["applied"], 0)

    def test_cut_out_drops_markers_inside_and_shifts_the_ones_after(self):
        out = replay_splices(self.x, [{"mode": "cut_out", "start": 10.0, "end": 30.0}], detected_markers=self.markers)
        times = [m["time"] for m in out["detected_markers"]]
        self.assertEqual(len(times), 3)                       # the marker at 20 s was inside the cut
        self.assertEqual(times[0], 5.0)
        shift = 30.01 - 10.0                                  # first sample past the cut minus the first one cut
        assert_allclose(times[1:], [55.5 - shift, 90.0 - shift], atol=1e-9)
        self.assertAlmostEqual(out["x"][-1], 99.99 - shift, places=9)
        self.assertEqual(out["applied"], 1)
        np.testing.assert_allclose(np.diff(out["x"]), 0.01, atol=1e-9)         # the timeline stays contiguous

    def test_keep_inside_keeps_absolute_times(self):
        out = replay_splices(self.x, [{"mode": "keep_inside", "start": 15.0, "end": 60.0}], detected_markers=self.markers)
        self.assertEqual([m["time"] for m in out["detected_markers"]], [20.0, 55.5])
        self.assertAlmostEqual(out["x"][0], 15.0)
        self.assertAlmostEqual(out["x"][-1], 60.0)

    def test_it_is_exactly_the_library_splice_functions_chained(self):
        splices = [{"mode": "cut_out", "start": 10.0, "end": 30.0},
                   {"mode": "keep_inside", "start": 20.0, "end": 60.0},
                   {"mode": "cut_out", "start": 40.0, "end": 42.5}]
        out = replay_splices(self.x, splices, detected_markers=self.markers)
        x, det = self.x, self.markers
        for s in splices:
            fn = splice_cut_out if s["mode"] == "cut_out" else splice_keep_inside
            r = fn(x, x, x, [], det, s["start"], s["end"])
            x, det = r["x"], r["detected_markers"]
        assert_array_equal(out["x"], x)
        self.assertEqual(out["detected_markers"], det)                        # bit-for-bit, not just close

    def test_a_splice_that_leaves_too_little_is_skipped_like_the_app_does(self):
        splices = [{"mode": "cut_out", "start": 0.0, "end": 99.995},           # would leave under 2 samples
                   {"mode": "cut_out", "start": 10.0, "end": 20.0}]
        out = replay_splices(self.x, splices, detected_markers=self.markers)
        self.assertEqual(out["applied"], 1)
        self.assertEqual(len(out["x"]), len(self.x) - 1001)

    def test_any_other_mode_reads_as_keep_inside(self):
        out = replay_splices(self.x, [{"mode": "whatever", "start": 15.0, "end": 60.0}], detected_markers=self.markers)
        self.assertEqual([m["time"] for m in out["detected_markers"]], [20.0, 55.5])

    def test_both_marker_lists_are_carried(self):
        user = [_marker(6.0, "Clap", None)]
        out = replay_splices(self.x, [{"mode": "cut_out", "start": 0.0, "end": 5.5}], markers=user,
                             detected_markers=self.markers)
        self.assertEqual(len(out["markers"]), 1)
        self.assertLess(out["markers"][0]["time"], 1.0)


def _fake_block(epocs, n=20_000, fs=1000.0, duration=None, blockname="Rat7-250101-101010"):
    """(read_block, calls) standing in for the tdt SDK. `epocs` = {store: (onsets, offsets)}; the recorded
    `calls` lists the evtype of every full-block read, so tests can tell whether streams were read."""
    calls = []
    t = np.arange(n) / fs
    y = (60 + 5 * np.sin(t)).astype(np.float32)

    heads = tdt.StructType()
    heads.stores = {f"{k}_": types.SimpleNamespace(type_str="epocs", name=f"{k}/") for k in epocs}
    heads.stores["x465A"] = types.SimpleNamespace(type_str="streams", name="x465A")

    data = tdt.StructType()
    data.streams = {"x465A": types.SimpleNamespace(data=y, fs=fs)}
    data.scalars = tdt.StructType()
    data.epocs = tdt.StructType()
    if duration is not None:
        data.info = types.SimpleNamespace(duration=datetime.timedelta(seconds=duration), blockname=blockname,
                                          start_date=datetime.datetime(2025, 12, 22, 21, 4, 8))

    def read_block(path, headers=None, evtype=None, store=None, verbose=None):
        if isinstance(headers, int) and headers == 1:
            return heads
        if store is not None:
            key = store.rstrip("/")
            onsets, offsets = epocs[key]
            block = tdt.StructType()
            block.epocs = tdt.StructType()
            block.epocs[f"{key}_"] = types.SimpleNamespace(name=key, onset=np.asarray(onsets, float),
                                                          offset=np.asarray(offsets, float))
            return block
        calls.append(list(evtype))
        if evtype and "streams" not in evtype:
            slim = tdt.StructType()
            slim.streams = tdt.StructType()
            slim.scalars = tdt.StructType()
            slim.epocs = tdt.StructType()
            if hasattr(data, "info"):
                slim.info = data.info
            return slim
        return data

    return read_block, calls


class TestScanTdtMarkers(unittest.TestCase):
    EPOCS = {"L1P": ([10.0, 12.0, 15.0], [10.2, 12.3, 15.1]), "Pmp": ([11.0], [13.0])}

    def test_lists_the_same_markers_as_get_event_markers_and_reads_no_streams(self):
        read_block, calls = _fake_block(self.EPOCS, duration=20.0)
        with mock.patch.object(tdt, "read_block", read_block):
            scan = P.scan_tdt_markers("ignored")
            full = P.get_event_markers(P.get_tdt_struct("ignored"))
        self.assertEqual(scan["markers"], full)
        self.assertEqual(len(scan["markers"]), 2 * (3 + 1))                    # onset + offset per event
        self.assertEqual(calls[0], ["scalars"])                                # the scan never asked for the streams
        self.assertEqual(calls[1], ["streams", "scalars"])                     # the ordinary read still does

    def test_span_and_block_details_come_from_the_block_info(self):
        read_block, _ = _fake_block(self.EPOCS, duration=20.0, blockname="PFC-GCaMP-Sucrose-251222-210404")
        with mock.patch.object(tdt, "read_block", read_block):
            scan = P.scan_tdt_markers("ignored")
        self.assertEqual(scan["t_range"], (0.0, 20.0))
        self.assertEqual(scan["block_name"], "PFC-GCaMP-Sucrose-251222-210404")
        self.assertEqual(scan["start_time"], "2025-12-22T21:04:08")
        self.assertEqual(scan["n_splices"], 0)

    def test_a_block_without_info_still_scans(self):
        read_block, _ = _fake_block(self.EPOCS)                                # no info attribute at all
        with mock.patch.object(tdt, "read_block", read_block):
            scan = P.scan_tdt_markers("ignored")
        self.assertIsNone(scan["t_range"])
        self.assertIsNone(scan["block_name"])
        self.assertIsNone(scan["start_time"])
        self.assertEqual(len(scan["markers"]), 8)

    def test_splices_are_replayed_on_the_times_and_the_exact_span_is_rebuilt(self):
        read_block, calls = _fake_block(self.EPOCS, n=20_000, fs=1000.0, duration=20.0)
        splices = [{"mode": "cut_out", "start": 11.5, "end": 14.0}]
        with mock.patch.object(tdt, "read_block", read_block):
            plain = P.scan_tdt_markers("ignored")
            cut = P.scan_tdt_markers("ignored", splices=splices)
        self.assertEqual(calls[-1], ["streams", "scalars"])                    # only spliced blocks need the time axis
        self.assertEqual(cut["n_splices"], 1)
        self.assertEqual(len(cut["markers"]), len(plain["markers"]) - 3)       # L1P 12.0 and 12.3, and Pmp 13.0, fell inside the cut
        t_before = sorted(m["time"] for m in plain["markers"])
        t_after = sorted(m["time"] for m in cut["markers"])
        self.assertEqual(t_before[:3], t_after[:3])                             # everything before the cut is untouched
        self.assertAlmostEqual(cut["t_range"][1], 19.999 - 2.501, places=6)    # the 2.5 s (+1 sample) cut shortens it
        self.assertEqual(cut["t_range"][0], 0.0)

    def test_progress_runs_from_the_start_to_a_hundred_percent(self):
        read_block, _ = _fake_block(self.EPOCS, duration=20.0)
        seen = []
        with mock.patch.object(tdt, "read_block", read_block):
            P.scan_tdt_markers("ignored", progress=lambda f, m: seen.append((f, m)))
        self.assertEqual(seen[-1][0], 1.0)
        self.assertTrue(all(a[0] <= b[0] for a, b in zip(seen, seen[1:])))
        self.assertTrue(all(m.isascii() for _, m in seen))


class TestGroupSpec(unittest.TestCase):
    def good(self, **kw):
        spec = group.GroupSpec(group_name="Sucrose",
                               subjects=[{"subject": "A", "folder": "a"}, {"subject": "B", "folder": "b"}],
                               markers=["L1P¹"])
        for k, v in kw.items():
            setattr(spec, k, v)
        return spec

    def test_defaults_follow_the_tdt_reference_example(self):
        spec = group.GroupSpec()
        self.assertEqual((spec.pre, spec.post), (10.0, 10.0))
        self.assertEqual(spec.baseline, (-10.0, -6.0))
        self.assertEqual(spec.response, (0.0, 10.0))
        self.assertEqual(spec.metrics, ["auc", "peak", "mean", "latency", "decay"])
        self.assertEqual((spec.signal, spec.peak_direction, spec.decay_fraction), ("dff", "absolute", 0.5))
        self.assertEqual((spec.smooth_seconds, spec.correction, spec.alpha), (0.5, "holm", 0.05))

    def test_a_good_spec_has_no_problems(self):
        self.assertEqual(self.good().problems(), [])

    def test_defaults_are_not_shared_between_instances(self):
        a, b = group.GroupSpec(), group.GroupSpec()
        a.metrics.append("extra")
        a.subjects.append({"subject": "x", "folder": "y"})
        self.assertEqual(b.metrics, list(group.METRICS))
        self.assertEqual(b.subjects, [])

    def test_json_round_trip(self):
        spec = self.good(baseline=(-8.0, -4.0), response=(0.5, 6.0), signal="zscore", peak_direction="negative",
                         metrics=["auc", "latency"], store_labels={"PP1_": "Left Lever"}, decay_fraction=0.37,
                         smooth_seconds=0.0, correction="bonferroni", alpha=0.01)
        text = json.dumps(spec.to_dict())
        back = group.GroupSpec.from_dict(json.loads(text))
        self.assertEqual(back, spec)
        self.assertIsInstance(back.baseline, tuple)

    def test_from_dict_ignores_keys_it_does_not_know(self):
        back = group.GroupSpec.from_dict({"group_name": "G", "some_future_option": 3})
        self.assertEqual(back.group_name, "G")

    def test_each_rule_is_reported(self):
        cases = {
            "one subject":            (self.good(subjects=[{"subject": "A", "folder": "a"}]), "at least two"),
            "blank subject name":     (self.good(subjects=[{"subject": " ", "folder": "a"}, {"subject": "B", "folder": "b"}]), "needs a name"),
            "repeated subject name":  (self.good(subjects=[{"subject": "A", "folder": "a"}, {"subject": "A", "folder": "b"}]), "unique"),
            "no marker":              (self.good(markers=[]), "at least one marker"),
            "repeated marker":        (self.good(markers=["x", "x"]), "only be picked once"),
            "zero pre":               (self.good(pre=0), "positive number"),
            "nan post":               (self.good(post=float("nan")), "positive number"),
            "baseline backwards":     (self.good(baseline=(-6.0, -10.0)), "baseline window must run"),
            "baseline after event":   (self.good(baseline=(-5.0, 2.0)), "baseline window must lie"),
            "baseline before window": (self.good(baseline=(-12.0, -6.0)), "baseline window must lie"),
            "response backwards":     (self.good(response=(5.0, 1.0)), "response window must run"),
            "response before event":  (self.good(response=(-1.0, 5.0)), "response window must lie"),
            "response past window":   (self.good(response=(0.0, 11.0)), "response window must lie"),
            "no metric":              (self.good(metrics=[]), "at least one measure"),
            "unknown metric":         (self.good(metrics=["auc", "median"]), "Unknown measure: median"),
            "bad signal":             (self.good(signal="raw"), "Signal must be"),
            "bad direction":          (self.good(peak_direction="up"), "Peak direction must be"),
            "decay of 100%":          (self.good(decay_fraction=1.0), "between 0 and 100%"),
            "bad regression":         (self.good(regression_method="lasso"), "Regression method must be"),
            "negative smoothing":     (self.good(smooth_seconds=-1), "smoothing window"),
            "bad correction":         (self.good(correction="sidak"), "Correction must be"),
            "alpha of 1":             (self.good(alpha=1.0), "significance level"),
        }
        for label, (spec, needle) in cases.items():
            with self.subTest(label):
                problems = spec.problems()
                self.assertTrue(any(needle in p for p in problems), problems)

    def test_window_problems_covers_only_the_window_rules(self):
        empty = group.GroupSpec()                                             # no subjects or markers, but a fine window
        self.assertEqual(empty.window_problems(), [])
        self.assertTrue(empty.problems())
        bad = group.GroupSpec(pre=5, baseline=(-10.0, -6.0))
        self.assertTrue(any("baseline window must lie" in p for p in bad.window_problems()))
        self.assertEqual(group.GroupSpec(pre=-1).window_problems(),
                         ["The window needs a positive number of seconds before and after the event."])

    def test_baseline_and_response_are_checked_against_the_chosen_window(self):
        self.assertEqual(self.good(pre=12, post=8, baseline=(-12.0, -6.0), response=(0.0, 8.0)).problems(), [])
        self.assertTrue(self.good(pre=5, post=8, baseline=(-10.0, -6.0)).problems())


class TestMarkerIndex(unittest.TestCase):
    SCANS = [
        {"subject": "A", "groups": {"L1P¹": [1, 2, 3], "Clap": [50], "Pmp¹": []}},
        {"subject": "B", "groups": {"L1P¹": [4, 5], "Clap": [60, 70]}},
        {"subject": "C", "groups": {"L1P¹": [9], "Sucrose": [1]}},
    ]

    def test_counts_per_marker_and_subject(self):
        idx = group.marker_index(self.SCANS)
        self.assertEqual(idx["L1P¹"], {"recordings": 3, "events": 6, "per_subject": {"A": 3, "B": 2, "C": 1}})
        self.assertEqual(idx["Clap"]["recordings"], 2)
        self.assertNotIn("Pmp¹", idx)                                     # an empty list is not a marker the recording has

    def test_common_markers_are_the_ones_every_recording_has(self):
        self.assertEqual(group.common_markers(self.SCANS), ["L1P¹"])
        self.assertEqual(group.common_markers(self.SCANS[:2]), ["Clap", "L1P¹"])
        self.assertEqual(group.common_markers([]), [])


class TestDesignSummary(unittest.TestCase):
    def scan(self, subject, times, t_range=(0.0, 100.0), marker="L1P"):
        return {"subject": subject, "groups": {marker: list(times)}, "t_range": t_range}

    def test_usable_means_the_whole_window_is_inside_the_recording(self):
        # window -10..+10: events at 5 and 95 have no full window; 10 and 90 sit exactly on the edge
        out = group.design_summary([self.scan("A", [5, 10, 50, 90, 95])], ["L1P"], 10, 10)
        cell = out["cells"]["A"]["L1P"]
        self.assertEqual((cell["events"], cell["usable"]), (5, 3))

    def test_unknown_recording_length_counts_every_event_as_usable(self):
        out = group.design_summary([self.scan("A", [1, 500], t_range=None)], ["L1P"], 10, 10)
        self.assertEqual(out["cells"]["A"]["L1P"]["usable"], 2)

    def test_overlap_is_windows_that_share_samples(self):
        # window length 20 s: gap 19.9 overlaps, gap 20 does not; the far event overlaps nothing
        out = group.design_summary([self.scan("A", [20, 39.9, 60, 80.5])], ["L1P"], 10, 10)
        cell = out["cells"]["A"]["L1P"]
        self.assertEqual(cell["usable"], 4)
        self.assertEqual(cell["overlapping"], 2)                               # 20 and 39.9 share; 60 / 80.5 are 20.1 and 20.5 apart

    def test_overlap_only_looks_at_usable_trials(self):
        out = group.design_summary([self.scan("A", [3, 12, 50])], ["L1P"], 10, 10)     # 3 and 12 are 9 s apart, 3 is unusable
        cell = out["cells"]["A"]["L1P"]
        self.assertEqual((cell["usable"], cell["overlapping"]), (2, 0))

    def test_events_are_sorted_before_they_are_compared(self):
        out = group.design_summary([self.scan("A", [60, 20, 30])], ["L1P"], 10, 10)
        self.assertEqual(out["cells"]["A"]["L1P"]["overlapping"], 2)

    def test_marker_totals_and_per_subject_range(self):
        scans = [self.scan("A", [20, 50, 80]), self.scan("B", [30]), self.scan("C", [])]
        out = group.design_summary(scans, ["L1P"], 10, 10)
        info = out["markers"]["L1P"]
        self.assertEqual((info["events"], info["usable"], info["subjects_with_trials"]), (4, 4, 2))
        self.assertEqual((info["min_usable"], info["max_usable"]), (0, 3))

    def test_a_marker_a_subject_lacks_is_a_zero_cell_and_a_warning(self):
        scans = [self.scan("A", [20, 60]), {"subject": "B", "groups": {}, "t_range": (0.0, 100.0)}]
        out = group.design_summary(scans, ["L1P"], 10, 10)
        self.assertEqual(out["cells"]["B"]["L1P"], {"events": 0, "usable": 0, "overlapping": 0})
        texts = [w["text"] for w in out["warnings"]]
        self.assertTrue(any("1 of 2 subjects have no usable trials" in t and "B" in t for t in texts), texts)

    def test_a_marker_nobody_has_is_flagged(self):
        out = group.design_summary([self.scan("A", [20]), self.scan("B", [30])], ["Clap"], 10, 10)
        self.assertTrue(any("no usable trials in any recording" in w["text"] for w in out["warnings"]))

    def test_heavy_overlap_is_a_warning_and_light_overlap_is_information(self):
        heavy = group.design_summary([self.scan("A", [20, 22, 24, 26])], ["L1P"], 10, 10)
        levels = {w["level"] for w in heavy["warnings"] if "share samples" in w["text"]}
        self.assertEqual(levels, {"warning"})
        light = group.design_summary([self.scan("A", [15, 20, 60, 90])], ["L1P"], 10, 10)     # 2 of 4 overlap = 50% -> warning
        light2 = group.design_summary([self.scan("A", [15, 20, 45, 70, 90])], ["L1P"], 10, 10)  # 2 of 5 = 40% -> information
        self.assertEqual([w["level"] for w in light["warnings"] if "share samples" in w["text"]], ["warning"])
        self.assertEqual([w["level"] for w in light2["warnings"] if "share samples" in w["text"]], ["info"])

    def test_few_subjects_warn_and_moderate_numbers_note_the_p_values(self):
        def scans(n):
            return [self.scan(f"S{i}", [20, 60]) for i in range(n)]
        few = group.design_summary(scans(4), ["L1P"], 10, 10)
        self.assertTrue(any(w["level"] == "warning" and "Only 4 subjects" in w["text"] for w in few["warnings"]))
        some = group.design_summary(scans(12), ["L1P"], 10, 10)
        self.assertTrue(any(w["level"] == "info" and "12 subjects" in w["text"] for w in some["warnings"]))
        many = group.design_summary(scans(25), ["L1P"], 10, 10)
        self.assertFalse(any("subjects" in w["text"] for w in many["warnings"]))

    def test_on_the_real_recording_shape(self):
        """The numbers measured on the test recording: 94-100% of trials overlap at the -10/+20 window."""
        rng = np.random.default_rng(3)
        ee1 = np.cumsum(10.5 + rng.normal(0, 0.4, 36)) + 60          # ~10.5 s apart, like EE1
        out = group.design_summary([self.scan("A", ee1, t_range=(0.0, 1037.8), marker="EE1")], ["EE1"], 10, 20)
        info = out["markers"]["EE1"]
        self.assertEqual(info["usable"], 36)
        self.assertGreaterEqual(info["overlapping"], 35)


if __name__ == "__main__":
    unittest.main()
