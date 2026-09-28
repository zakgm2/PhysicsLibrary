"""
progress.py
-----------
Progress reporting for the library's slow operations, built on tqdm.

Every slow public function takes an optional `progress` argument:

    progress=None       silent (the default) - costs nothing
    progress=True       a tqdm progress bar on the console, for scripts and notebooks
    progress=callable   called as progress(fraction, message): fraction runs 0.0 -> 1.0
                        and message says what is happening right now. This is how a GUI
                        drives a progress toast; the callable may be called from whatever
                        thread the work runs on.

Inside a function the work is split into named stages with relative weights (`Plan`).
A loop inside a stage is wrapped with `plan.track(...)`, a tqdm iterator under the hood, so
updates are throttled by tqdm (at most ~20 a second) instead of firing on every iteration.
A stage can be handed to another function as its own `progress` with `plan.sub(...)`, so
progress composes across nested calls and always reads as one 0-100% run.

    plan = Plan(progress, [("Reading files", 30), ("Fitting", 70)])
    for path in plan.track("Reading files", paths):
        ...
    fit(data, progress=plan.sub("Fitting"))

Progress never changes a result, and a callback that raises is switched off with a warning
rather than aborting the computation it was only watching.
"""

import contextlib
import sys
import threading
import warnings

from tqdm import tqdm

__all__ = ["Plan", "track", "coerce"]


class _NullFile:
    """Swallows tqdm's console output: the bars below only report numbers."""

    def write(self, *_):
        return 0

    def flush(self):
        pass


_NULL_FILE = _NullFile()


class _CallbackBar(tqdm):
    """A tqdm bar that draws nothing and hands its fraction to a callback instead.

    It still gets everything tqdm is good at: iterating any iterable, the update()/total
    bookkeeping, and throttling (mininterval), so a fast loop reports ~20 times a second at
    most however many iterations it runs. It has its own lock and no watchdog thread, so it
    never touches the state of any other tqdm bar in the process.
    """

    monitor_interval = 0
    _lock = threading.RLock()

    def __init__(self, *args, callback=None, **kwargs):
        self._callback = callback          # must exist before tqdm's __init__ draws the first frame
        kwargs.setdefault("file", _NULL_FILE)
        kwargs.setdefault("disable", False)
        kwargs.setdefault("leave", False)
        kwargs.setdefault("mininterval", 0.05)
        kwargs.setdefault("miniters", 1)
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        if self._callback is not None and self.total:
            self._callback(min(1.0, self.n / self.total))
        return True


class _ConsoleSink:
    """progress=True: one 0-100% tqdm bar on stderr for the whole run."""

    def __init__(self):
        stream = sys.stderr if sys.stderr is not None else _NULL_FILE   # windowed frozen apps have no stderr
        self._bar = tqdm(total=100, file=stream, leave=True, mininterval=0.1,
                         bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}")

    def __call__(self, fraction, message):
        bar = self._bar
        bar.set_description_str(message, refresh=False)
        bar.update(fraction * 100 - bar.n)
        if fraction >= 1.0:
            bar.close()


def coerce(progress):
    """None / False -> None (silent); True -> a console tqdm bar; a callable -> itself."""
    if progress is None or progress is False:
        return None
    if progress is True:
        return _ConsoleSink()
    if callable(progress):
        return progress
    raise TypeError("progress must be None, True, or a callable(fraction, message); "
                    f"got {type(progress).__name__}")


class _NoBar:
    """Stands in for a bar when nobody is listening, so `bar.update(n)` needs no `if`."""

    def update(self, n=1):
        pass


class Plan:
    """Turns named, weighted stages (and the loops inside them) into one 0.0-1.0 fraction.

    Parameters
    ----------
    progress : None, True or callable(fraction, message)
        Where the overall progress goes; see the module docstring. With None every method
        below is a near no-op and track() hands the iterable straight back.
    stages : list of (name, weight)
        The steps of the operation in order; a stage covers weight / sum(weights) of the bar.
        Weights are only relative, so estimate them from how long each step takes.
    """

    def __init__(self, progress, stages):
        self._sink = coerce(progress)
        total = float(sum(weight for _, weight in stages)) or 1.0
        self._ranges = {}
        lo = 0.0
        for name, weight in stages:
            hi = lo + weight / total
            self._ranges[name] = (lo, hi)
            lo = hi
        self._last = 0.0

    @property
    def active(self):
        """True if anyone is listening (so callers can skip work only done for reporting)."""
        return self._sink is not None

    def _report(self, fraction, message):
        sink = self._sink
        if sink is None:
            return
        fraction = min(1.0, max(self._last, float(fraction)))     # never backwards, never past 100%
        self._last = fraction
        try:
            sink(fraction, message or "")
        except Exception as exc:                                  # a broken listener must not kill the work
            self._sink = None
            warnings.warn(f"progress callback failed and was turned off: {exc!r}",
                          RuntimeWarning, stacklevel=3)

    def begin(self, name, message=None):
        """Mark the start of a stage (for a step with no loop inside it to tick)."""
        self._report(self._ranges[name][0], message or name)

    def track(self, name, iterable, total=None, message=None):
        """Iterate `iterable` as stage `name`, reporting through the stage's slice of the bar."""
        if self._sink is None:
            return iterable
        return self._track(name, iterable, total, message or name)

    def _track(self, name, iterable, total, text):
        lo, hi = self._ranges[name]
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        if not total:                                             # unknown length: just mark both ends
            self._report(lo, text)
            yield from iterable
            self._report(hi, text)
            return
        bar = _CallbackBar(iterable, total=total,
                           callback=lambda fraction: self._report(lo + (hi - lo) * fraction, text))
        try:
            yield from bar
        finally:
            bar.close()
        self._report(hi, text)                                    # skipped if the loop was abandoned

    @contextlib.contextmanager
    def bar(self, name, total, message=None):
        """Stage `name` driven by hand: `with plan.bar("Reading", size) as bar: bar.update(n_bytes)`."""
        if self._sink is None or not total:
            yield _NoBar()
            return
        lo, hi = self._ranges[name]
        text = message or name
        bar = _CallbackBar(total=total, callback=lambda fraction: self._report(lo + (hi - lo) * fraction, text))
        try:
            yield bar
        finally:
            bar.close()
        self._report(hi, text)

    def sub(self, name):
        """Stage `name` as a `progress` callback of its own, for handing to another function.

        The callee reports 0.0 -> 1.0 and its messages; both are mapped into this stage's slice
        of the bar. Returns None when nobody is listening, so the callee stays silent too.
        """
        if self._sink is None:
            return None
        lo, hi = self._ranges[name]

        def report(fraction, message=""):
            self._report(lo + (hi - lo) * min(1.0, max(0.0, fraction)), message or name)

        return report

    def done(self, message=""):
        """Report 100%."""
        self._report(1.0, message)


def track(iterable, progress=None, description="Working", total=None):
    """Iterate `iterable` with progress and no stages: `for f in track(files, progress, "Reading files")`."""
    return Plan(progress, [(description, 1)]).track(description, iterable, total=total)
