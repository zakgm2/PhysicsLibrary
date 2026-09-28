"""Progress reporting: the Plan / tqdm machinery itself, and that every instrumented function
reports sensibly while returning exactly what it returns without progress.

The heavy inputs are faked: a TDT block (tdt.read_block is patched), a synthetic Oxysoft export,
generic tabular files, and a stub in place of sentence-transformers."""

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
import warnings
import zlib
from unittest import mock

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import tdt

from PhysicsLibrary import processing_TDT as P
from PhysicsLibrary.analysis.event_peth import compute_event_zscore_peth
from PhysicsLibrary.analysis.peak_finder import find_peak_near_events
from PhysicsLibrary.file_parser import load_dataset, load_dataset_file
from PhysicsLibrary.file_parser_generic import load_any_file
from PhysicsLibrary.field_study_validation import run_validation_pipeline
from PhysicsLibrary.loaders.oxysoft_loader import load_oxysoft, load_oxysoft_file
from PhysicsLibrary.progress import Plan, coerce, track
from PhysicsLibrary.text_field_study import run_field_study_pipeline


class Recorder:
    """A progress callback that keeps everything it is told."""

    def __init__(self):
        self.events = []
        self.threads = set()

    def __call__(self, fraction, message):
        self.events.append((fraction, message))
        self.threads.add(threading.get_ident())

    @property
    def fractions(self):
        return [f for f, _ in self.events]

    @property
    def messages(self):
        return {m for _, m in self.events}


class ProgressAssertions(unittest.TestCase):
    def assert_sane(self, rec, must_finish=True):
        """Non-empty, inside 0..1, never backwards, and (normally) ending at exactly 100%."""
        self.assertTrue(rec.events, "nothing was reported")
        f = rec.fractions
        self.assertTrue(all(0.0 <= x <= 1.0 for x in f), f)
        self.assertTrue(all(a <= b for a, b in zip(f, f[1:])), f"went backwards: {f}")
        # Plain ASCII, so a script can print them to any console or log (a redirected Windows
        # stdout is cp1252 and raises on a character like the Greek delta).
        self.assertTrue(all(m.isascii() for m in rec.messages), rec.messages)
        if must_finish:
            self.assertEqual(f[-1], 1.0)


class TestPlan(ProgressAssertions):
    def test_stages_split_the_bar_by_weight(self):
        rec = Recorder()
        plan = Plan(rec, [("a", 1), ("b", 3)])
        plan.begin("a")
        plan.begin("b")
        plan.done()
        self.assertEqual(rec.fractions, [0.0, 0.25, 1.0])
        self.assertEqual([m for _, m in rec.events], ["a", "b", ""])

    def test_a_loop_ticks_inside_its_stage_and_yields_every_item(self):
        rec = Recorder()
        plan = Plan(rec, [("read", 1), ("fit", 1)])
        seen = list(plan.track("read", range(10)))
        self.assertEqual(seen, list(range(10)))
        self.assertTrue(all(f <= 0.5 for f in rec.fractions))
        self.assertEqual(rec.fractions[-1], 0.5)                  # the stage ends exactly at its boundary
        self.assert_sane(rec, must_finish=False)

    def test_a_sub_progress_is_mapped_into_the_stage(self):
        rec = Recorder()
        plan = Plan(rec, [("a", 1), ("b", 3)])
        sub = plan.sub("b")
        sub(0.0, "start")
        sub(0.5, "half")
        sub(1.0, "end")
        self.assertEqual(rec.fractions, [0.25, 0.625, 1.0])
        self.assertEqual([m for _, m in rec.events], ["start", "half", "end"])

    def test_a_sub_progress_without_a_message_falls_back_to_the_stage_name(self):
        rec = Recorder()
        Plan(rec, [("only", 1)]).sub("only")(0.5)
        self.assertEqual(rec.events, [(0.5, "only")])

    def test_manual_bar_updates(self):
        rec = Recorder()
        plan = Plan(rec, [("bytes", 1)])
        with plan.bar("bytes", total=1000) as bar:
            for _ in range(10):
                bar.update(100)
        self.assert_sane(rec)

    def test_iterables_of_unknown_length_report_both_ends(self):
        rec = Recorder()
        out = list(Plan(rec, [("s", 1)]).track("s", (i for i in range(5))))
        self.assertEqual(out, [0, 1, 2, 3, 4])
        self.assertEqual(rec.fractions, [0.0, 1.0])

    def test_an_empty_iterable_still_completes_its_stage(self):
        rec = Recorder()
        self.assertEqual(list(Plan(rec, [("s", 1)]).track("s", [])), [])
        self.assertEqual(rec.fractions[-1], 1.0)

    def test_never_backwards_and_never_past_one(self):
        rec = Recorder()
        sub = Plan(rec, [("s", 1)]).sub("s")
        sub(0.6)
        sub(0.2)        # a stale value from a callee
        sub(7.0)
        self.assertEqual(rec.fractions, [0.6, 0.6, 1.0])

    def test_silent_when_progress_is_none(self):
        plan = Plan(None, [("s", 1)])
        data = [1, 2, 3]
        self.assertIs(plan.track("s", data), data)                # handed straight back: zero overhead
        self.assertIsNone(plan.sub("s"))                          # so callees stay silent too
        self.assertFalse(plan.active)
        plan.begin("s")
        plan.done()                                               # no-ops
        with plan.bar("s", total=10) as bar:
            bar.update(5)
        self.assertIs(track(data, None), data)

    def test_the_track_helper(self):
        rec = Recorder()
        self.assertEqual(list(track(range(4), rec, "Doing it")), [0, 1, 2, 3])
        self.assertIn("Doing it", rec.messages)
        self.assert_sane(rec)

    def test_a_callback_that_raises_is_switched_off_and_the_work_carries_on(self):
        calls = []

        def broken(fraction, message):
            calls.append(fraction)
            raise RuntimeError("boom")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = list(track(range(5), broken, "x"))
        self.assertEqual(out, [0, 1, 2, 3, 4])
        self.assertEqual(len(calls), 1)                           # never called again after failing
        self.assertTrue(any("progress callback failed" in str(w.message) for w in caught))

    def test_an_abandoned_loop_does_not_claim_completion(self):
        rec = Recorder()
        for i in Plan(rec, [("s", 1)]).track("s", range(100)):
            if i == 3:
                break
        self.assertLess(max(rec.fractions), 1.0)

    def test_a_loop_that_raises_does_not_claim_completion(self):
        rec = Recorder()
        with self.assertRaises(ValueError):
            for i in Plan(rec, [("s", 1)]).track("s", range(10)):
                if i == 4:
                    raise ValueError
        self.assertLess(max(rec.fractions), 1.0)

    def test_reports_are_throttled_not_one_per_iteration(self):
        rec = Recorder()
        for _ in track(range(300_000), rec, "big"):
            pass
        self.assert_sane(rec)
        self.assertLess(len(rec.events), 1000)                    # tqdm's mininterval, not 300,000 calls

    def test_no_console_output_and_no_extra_threads(self):
        before = {t.name for t in threading.enumerate()}
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            for _ in track(range(200), Recorder(), "quiet"):
                pass
        self.assertEqual(buf.getvalue(), "")
        self.assertEqual({t.name for t in threading.enumerate()}, before)     # tqdm's monitor thread stays off

    def test_works_from_another_thread(self):
        rec = Recorder()
        worker = threading.Thread(target=lambda: list(track(range(50), rec, "bg")))
        worker.start()
        worker.join()
        self.assert_sane(rec)
        self.assertNotIn(threading.get_ident(), rec.threads)      # reported from the worker, as a GUI needs to handle

    def test_console_mode_draws_a_tqdm_bar(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            plan = Plan(True, [("a", 1), ("b", 1)])
            list(plan.track("a", range(3)))
            plan.begin("b", "second half")
            plan.done("finished")
        text = buf.getvalue()
        self.assertIn("100%", text)
        self.assertIn("finished", text)

    def test_console_mode_survives_a_missing_stderr(self):
        with mock.patch.object(sys, "stderr", None):              # what a windowed frozen app has
            plan = Plan(True, [("a", 1)])
            list(plan.track("a", range(3)))
            plan.done()

    def test_coerce(self):
        self.assertIsNone(coerce(None))
        self.assertIsNone(coerce(False))
        fn = lambda f, m: None
        self.assertIs(coerce(fn), fn)
        with self.assertRaises(TypeError):
            coerce("yes please")


# ---------------------------------------------------------------------------------------------
# a fake TDT block
# ---------------------------------------------------------------------------------------------

def _fake_tdt(n=20_000, fs=1000.0, n_epocs=5, seed=1):
    """(read_block, heads, data) standing in for the tdt SDK: two streams and `n_epocs` epoc stores."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    base = 60 + 30 * np.exp(-t / 8)
    motion = 1 + 0.02 * np.sin(t)
    f415 = (0.5 * base * motion + rng.normal(0, 0.1, n)).astype(np.float32)
    f465 = (base * motion * (1 + 0.1 * np.exp(-0.5 * ((t - 10) / 0.5) ** 2)) + rng.normal(0, 0.1, n)).astype(np.float32)

    heads = tdt.StructType()
    heads.stores = {f"EP{i}_": types.SimpleNamespace(type_str="epocs", name=f"EP{i}/") for i in range(n_epocs)}
    heads.stores["x465A"] = types.SimpleNamespace(type_str="streams", name="x465A")

    data = tdt.StructType()
    data.streams = {"x465A": types.SimpleNamespace(data=f465, fs=fs), "x415A": types.SimpleNamespace(data=f415, fs=fs)}
    data.scalars = tdt.StructType()
    data.epocs = tdt.StructType()

    def epoc(i):
        return types.SimpleNamespace(name=f"EP{i}", onset=np.array([3.0 + i, 6.0 + i]), offset=np.array([4.0 + i, 7.0 + i]))

    def read_block(path, headers=None, evtype=None, store=None, verbose=None):
        if isinstance(headers, int) and headers == 1:
            return heads
        if store is not None:
            i = int(store[2:].rstrip("/"))
            block = tdt.StructType()
            block.epocs = tdt.StructType()
            block.epocs[f"EP{i}_"] = epoc(i)
            return block
        return data

    return read_block, heads, data


class TestTDTPipeline(ProgressAssertions):
    def test_get_tdt_struct_reports_each_epoc_store_and_returns_the_same_block(self):
        read_block, _, _ = _fake_tdt(n_epocs=5)
        with mock.patch.object(tdt, "read_block", read_block):
            plain = P.get_tdt_struct("ignored")
            rec = Recorder()
            watched = P.get_tdt_struct("ignored", progress=rec)
        self.assert_sane(rec)
        self.assertIn("Reading streams", rec.messages)
        self.assertIn("Reading event stores", rec.messages)
        self.assertEqual(sorted(plain.epocs.keys()), sorted(watched.epocs.keys()))
        self.assertEqual(len(watched.epocs.keys()), 5)

    def test_compute_dff_reports_its_stages_and_the_result_is_unchanged(self):
        read_block, _, data = _fake_tdt()
        y465 = data.streams["x465A"].data
        y415 = data.streams["x415A"].data
        plain = P.compute_dff(y465, y415, 1000.0, "ols")
        rec = Recorder()
        watched = P.compute_dff(y465, y415, 1000.0, "ols", progress=rec)
        self.assert_sane(rec)
        self.assertTrue({"Fitting motion correction", "Fitting photobleaching baseline",
                         "Normalising and filtering"} <= rec.messages)
        for key in ("raw", "dff", "f0"):
            assert_allclose(plain[key], watched[key])

    def test_compute_dff_without_a_415_stream_has_no_motion_correction_stage(self):
        _, _, data = _fake_tdt()
        rec = Recorder()
        P.compute_dff(data.streams["x465A"].data, None, 1000.0, progress=rec)
        self.assert_sane(rec)
        self.assertNotIn("Fitting motion correction", rec.messages)

    def test_process_tdt_folder_reports_one_run_from_zero_to_a_hundred_percent(self):
        read_block, _, _ = _fake_tdt(n_epocs=6)
        with mock.patch.object(tdt, "read_block", read_block):
            plain = P.process_tdt_folder("ignored", regression_method="ols")
            rec = Recorder()
            watched = P.process_tdt_folder("ignored", regression_method="ols", progress=rec)
        self.assert_sane(rec)
        self.assertEqual(rec.fractions[0], 0.0)
        self.assertTrue({"Reading streams", "Reading event stores", "Fitting photobleaching baseline",
                         "Extracting markers"} <= rec.messages)
        self.assertGreater(len(rec.events), 6)
        assert_allclose(plain["corr"], watched["corr"])
        assert_allclose(plain["x"], watched["x"])
        self.assertEqual(len(plain["markers"]), len(watched["markers"]))

    def test_progress_true_draws_a_console_bar(self):
        read_block, _, _ = _fake_tdt(n_epocs=2)
        buf = io.StringIO()
        with mock.patch.object(tdt, "read_block", read_block), contextlib.redirect_stderr(buf):
            P.process_tdt_folder("ignored", progress=True)
        self.assertIn("100%", buf.getvalue())

    def test_load_dataset_passes_progress_through_to_a_tdt_load(self):
        read_block, _, _ = _fake_tdt(n_epocs=2)
        rec = Recorder()
        with mock.patch.object(tdt, "read_block", read_block), \
                mock.patch("PhysicsLibrary.loaders.tdt_loader.processing_TDT.validate_tdt_folder", return_value=(True, "ok")):
            ds = load_dataset("ignored", fmt=__import__("PhysicsLibrary").DataFormat.TDT, progress=rec)
        self.assertEqual(ds.source_format, "TDT")
        self.assert_sane(rec)


# ---------------------------------------------------------------------------------------------
# Oxysoft and generic files
# ---------------------------------------------------------------------------------------------

def _write_oxysoft(path, n_rows, seed=0):
    rng = np.random.default_rng(seed)
    lines = [
        "Some header\tvalue",
        "Datafile sample rate\t10 Hz",         # also what detect_format_file looks for
        "Legend:",
        "1\tsample number",
        "2\tRx1 - Tx1 O2Hb (file)",
        "3\tRx1 - Tx1 HHb (file)",
        "4\tRx1 - Tx2 O2Hb (file)",
        "5\tRx1 - Tx2 HHb (file)",
        "6\tEvent (Event)",
        "1\t2\t3\t4\t5\t6",
        "0\t0\t0\t0\t0\t0",
    ]
    for i in range(1, n_rows + 1):
        vals = rng.normal(0, 1, 4)
        event = "1" if i % 5000 == 0 else "0"
        lines.append("\t".join([str(i)] + [f"{v:.6f}" for v in vals] + [event]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


class TestFileLoaders(ProgressAssertions):
    def test_oxysoft_file_progress_follows_the_bytes_read_and_the_result_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "big.txt")
            _write_oxysoft(path, 60_000)                          # ~2.5 MB: several progress chunks
            plain = load_oxysoft_file(path)
            rec = Recorder()
            watched = load_oxysoft_file(path, progress=rec)
            via_dispatch = Recorder()
            load_dataset_file(path, progress=via_dispatch)
        self.assert_sane(rec)
        self.assertGreaterEqual(len(rec.events), 3)
        self.assertTrue({"Reading file", "Building channels"} <= rec.messages)
        assert_allclose(plain.signals, watched.signals)
        self.assertEqual(plain.events, watched.events)
        self.assert_sane(via_dispatch)

    def test_oxysoft_folder_progress_is_weighted_by_file_size(self):
        with tempfile.TemporaryDirectory() as d:
            _write_oxysoft(os.path.join(d, "a.txt"), 30_000, seed=1)
            _write_oxysoft(os.path.join(d, "b.txt"), 3_000, seed=2)
            plain = load_oxysoft(d, "folder")
            rec = Recorder()
            watched = load_oxysoft(d, "folder", progress=rec)
        self.assert_sane(rec)
        self.assertIn("Combining files", rec.messages)
        assert_allclose(plain.signals, watched.signals)
        # the big file (a.txt) is ~90% of the work, so the bar is well past half before b.txt starts
        first_b = next(i for i, (_, m) in enumerate(rec.events) if m == "b.txt")
        self.assertGreater(rec.fractions[first_b], 0.7)

    def test_generic_csv_reports_both_conversion_passes(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "table.csv")
            with open(path, "w") as fh:
                fh.write("time,a,b\n")
                for i in range(30_000):
                    fh.write(f"{i * 0.01:.2f},{rng.normal():.5f},{rng.normal():.5f}\n")
            plain = load_any_file(path)
            rec = Recorder()
            watched = load_any_file(path, progress=rec)
            tsv = os.path.join(d, "table.tsv")
            with open(path) as src, open(tsv, "w") as dst:
                dst.write(src.read().replace(",", "\t"))
            tsv_rec = Recorder()
            load_any_file(tsv, progress=tsv_rec)
        self.assert_sane(rec)
        self.assertTrue({"Reading file", "Finding the header row", "Converting to numbers"} <= rec.messages)
        assert_allclose(plain[0].data, watched[0].data)
        self.assertEqual(plain[0].headers, watched[0].headers)
        self.assert_sane(tsv_rec)

    def test_generic_excel_reports_per_sheet(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "book.xlsx")
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "first"
            ws.append(["t", "y"])
            for i in range(50):
                ws.append([i, i * 2.0])
            ws2 = wb.create_sheet("second")
            ws2.append(["t", "z"])
            for i in range(50):
                ws2.append([i, i * 3.0])
            wb.save(path)
            plain = load_any_file(path)
            rec = Recorder()
            watched = load_any_file(path, progress=rec)
        self.assert_sane(rec)
        self.assertTrue({"Opening workbook", "Reading sheets"} <= rec.messages)
        self.assertEqual([t.name for t in plain], [t.name for t in watched])
        assert_allclose(plain[1].data, watched[1].data)


# ---------------------------------------------------------------------------------------------
# analysis loops
# ---------------------------------------------------------------------------------------------

class TestAnalysisLoops(ProgressAssertions):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.t = np.arange(0, 400, 1 / 20.0)
        self.y = rng.normal(0, 0.05, self.t.size)
        for ev in (100, 200, 300):
            self.y += 1.0 * np.exp(-0.5 * ((self.t - (ev + 2.0)) / 0.7) ** 2)
        self.events = [100, 200, 300]

    def test_event_peth_ticks_per_event_and_is_unchanged(self):
        plain = compute_event_zscore_peth(self.t, self.y, self.events, pre=10, post=10)
        rec = Recorder()
        watched = compute_event_zscore_peth(self.t, self.y, self.events, pre=10, post=10, progress=rec)
        self.assert_sane(rec)
        assert_allclose(plain["trial_matrix"], watched["trial_matrix"])
        self.assertEqual(plain["trial_event_times"], watched["trial_event_times"])

    def test_peak_search_ticks_per_event_and_is_unchanged(self):
        plain = find_peak_near_events(self.t, self.y, self.events, pre=5, post=10, z_threshold=5.0)
        rec = Recorder()
        watched = find_peak_near_events(self.t, self.y, self.events, pre=5, post=10, z_threshold=5.0, progress=rec)
        self.assert_sane(rec)
        self.assertEqual(plain, watched)


# ---------------------------------------------------------------------------------------------
# the text-study pipelines, with a stand-in for sentence-transformers
# ---------------------------------------------------------------------------------------------

class _StubModel:
    """Deterministic bag-of-hashed-words embeddings, L2-normalised: no torch, no download."""

    def __init__(self, name):
        self.name = name

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), 24))
        for i, text in enumerate(texts):
            for word in str(text).lower().split():
                out[i, zlib.crc32(word.encode()) % 24] += 1.0
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


def _write_study(folder, n=14, seed=7):
    rng = np.random.default_rng(seed)
    vocab = "apple river stone cloud bright quiet garden window paper engine silver forest morning".split()
    for i in range(n):
        subject = {"participant_id": f"P{i:02d}"}
        core = list(rng.choice(vocab, 4))                         # shared within a subject, so pairs correlate
        for q in ("q1", "q2", "q3"):
            subject[q] = " ".join(core + list(rng.choice(vocab, rng.integers(3, 9))))
        with open(os.path.join(folder, f"P-{i:02d}.json"), "w", encoding="utf-8") as fh:
            json.dump(subject, fh)


class TestTextStudyPipelines(ProgressAssertions):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        _write_study(cls._tmp.name)
        cls.folder = cls._tmp.name
        stub = types.SimpleNamespace(SentenceTransformer=_StubModel)
        cls._patch = mock.patch.dict(sys.modules, {"sentence_transformers": stub})
        cls._patch.start()

    @classmethod
    def tearDownClass(cls):
        cls._patch.stop()
        cls._tmp.cleanup()

    def test_field_study_pipeline_progress_and_unchanged_result(self):
        kwargs = dict(text_fields=["q1", "q2", "q3"], delta_pair=("q1", "q3"),
                      paired_fields=[("q1", "q2", "pair12"), ("q2", "q3", "pair23")], n_null=40, rng_seed=1)
        plain = run_field_study_pipeline(self.folder, **kwargs)
        rec = Recorder()
        watched = run_field_study_pipeline(self.folder, progress=rec, **kwargs)
        self.assert_sane(rec)
        self.assertTrue({"Reading study files", "Loading language model", "Embedding responses",
                         "Comparing field pairs", "Checking word-count confounds"} <= rec.messages)
        pd.testing.assert_frame_equal(plain, watched)

    def test_field_study_pipeline_without_pairs_still_finishes(self):
        rec = Recorder()
        run_field_study_pipeline(self.folder, text_fields=["q1"], progress=rec)
        self.assert_sane(rec)

    def test_validation_pipeline_progress_and_unchanged_result(self):
        kwargs = dict(text_fields=["q1", "q2"], paired_fields=[("q1", "q2", "pair12")],
                      n_null=40, n_boot=40, rng_seed=1)
        plain = run_validation_pipeline(self.folder, **kwargs)
        rec = Recorder()
        watched = run_validation_pipeline(self.folder, progress=rec, **kwargs)
        self.assert_sane(rec)
        self.assertTrue({"Loading language model", "Embedding responses", "Validating field pairs"} <= rec.messages)
        pd.testing.assert_frame_equal(plain, watched)


if __name__ == "__main__":
    unittest.main()
