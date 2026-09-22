# PhysicsLibrary API Reference

Data processing and analysis library for TDT fibre-photometry, Oxysoft
NIRS, Terranova EFNMR, generic tabular data, and text-field studies.
Everything below is importable directly off the top-level package:

```python
import PhysicsLibrary as pl
pl.process\_tdt\_folder(folder\_path)
```

Install: `pip install ZaksPhysicsLibrary`

\---

## Contents

* [Loading Data](#loading-data)
* [TDT Photometry Processing](#tdt-photometry-processing)
* [Signal Analysis](#signal-analysis)

  * [General helpers](#general-helpers)
  * [Z-score / event-triggered analysis](#z-score--event-triggered-analysis)
  * [FFT](#fft)
  * [Curve fitting](#curve-fitting)
  * [Peak detection](#peak-detection)
  * [Area under the curve](#area-under-the-curve)
  * [Peri-event statistics](#peri-event-statistics)
* [Non-destructive splicing](#non-destructive-splicing)
* [Curve-fit models](#curve-fit-models)
* [Text Field Study](#text-field-study)
* [Text Field Study — Statistical Validation](#text-field-study--statistical-validation)
* [Optional dependencies](#optional-dependencies)

\---

## Loading Data

Everything here returns a `Dataset` — a universal container regardless of
source format — except `load\_any\_file` (tabular files, which can hold
several independent tables) and `load\_pt2` (a raw image array).

### `Dataset`

The struct every loader returns:

```python
@dataclass
class Dataset:
    source\_format: str          # 'TDT' | 'Oxysoft'
    folder\_path:   str
    folder\_name:   str

    sample\_rate: float = 0.0    # Hz
    num\_samples: int   = 0
    duration\_s:  float = 0.0    # seconds

    signals: Optional\[np.ndarray] = None   # shape (num\_channels, num\_samples)

    channel\_names: list\[str] = field(default\_factory=list)
    num\_channels:  int       = 0

    events: list\[dict] = field(default\_factory=list)   # {'label': str, 'sample': int}
    metadata: dict = field(default\_factory=dict)        # format-specific raw header blob
```

### `DataFormat`

`Enum` with members `TDT`, `OXYSOFT`, `UNKNOWN`.

### `detect\_format(folder\_path: str) -> DataFormat`

Inspects a folder's contents and returns the matching format. TDT is
checked first (proprietary extensions), then Oxysoft (`.txt` marker).

### `detect\_format\_file(file\_path: str) -> DataFormat`

Same, for a single file rather than a folder. Currently only recognizes
Oxysoft `.txt` exports.

### `load\_dataset(folder\_path, fmt=None, regression\_method="ransac") -> Dataset`

Top-level dispatcher — detects the format (unless `fmt` is given
explicitly) and routes to the matching loader.

* `regression\_method` (`"ransac" | "huber" | "ols"`): TDT-only, forwarded
to the motion-correction step (see [`process\_tdt\_folder`](#process_tdt_folderfolder_path-regression_methodransac)). Ignored for Oxysoft, which has no
such correction step.

### `load\_dataset\_file(file\_path: str) -> Dataset`

Loads a single file (rather than a folder) into a `Dataset`. Currently
supports Oxysoft `.txt` exports.

### `load\_any\_file(path: str) -> list\[GenericTable]`

Best-effort parser for arbitrary tabular data — `.xlsx`/`.xls` (needs
`openpyxl`), `.csv`, `.tsv`, or `.txt`/`.dat` with a sniffed delimiter.
Detects **multiple independent sub-tables** within one sheet/file (laid
out side-by-side or stacked, separated by empty rows/columns) and
returns each as its own table so a caller can pick which one to use.
Raises `ValueError` if nothing usable is found.

Each result is a `GenericTable`:

```python
@dataclass
class GenericTable:
    name:     str
    headers:  list\[str]
    data:     np.ndarray   # (n\_rows, n\_cols) float64, NaN for missing
    metadata: dict = field(default\_factory=dict)
```

### `load\_pt2(path: str) -> np.ndarray`

Parses a Terranova/Prospa `.pt2` 2D NMR/MRI image file (magnitude image
stored as little-endian float32 after a `LAER` marker). Returns a 2D
`(n, n)` float32 array.

> `load\_oxysoft`/`load\_oxysoft\_file`/`load\_tdt` exist as the actual
> per-format parsers behind `load\_dataset`, but aren't re-exported at
> the top level — reach them via `load\_dataset`/`load\_dataset\_file`
> unless you specifically need to bypass format auto-detection.

\---

## TDT Photometry Processing

### `process\_tdt\_folder(folder\_path, regression\_method="ransac")`

The full photometry pipeline, and the main TDT entry point: load block →
extract 465nm signal (+ optional 415nm isosbestic reference) → motion
correction → bleaching correction → ΔF/F → denoise → event markers.

* `regression\_method` (`"ransac" | "huber" | "ols"`, see
[`REGRESSION\_METHODS`](#regression_methods) below) — which regression the
isosbestic-vs-signal motion correction uses.

**Returns** `dict`:

|key|meaning|
|-|-|
|`x`|time vector|
|`raw`|corrected fluorescence signal|
|`corr` / `dff`|final ΔF/F (denoised) — `dff` is an alias for `corr`|
|`f0`|bleaching baseline|
|`fs`|sampling frequency|
|`store`|signal label|
|`markers`|behavioral event markers (see [`get\_event\_markers`](#get_event_markersdata))|
|`channels`|`\[{"key", "label", "y"}, ...]` — raw, un-corrected per-wavelength streams (`main\_driver`, plus `isosbestic` if a 415 reference exists)|
|`motion\_correction\_inlier\_fraction`|fraction (0–1) of samples kept as inliers during motion correction, or `None` if there was no 415 reference stream|

### `REGRESSION\_METHODS`

```python
REGRESSION\_METHODS = ("ransac", "huber", "ols")
```

The three motion-correction regressions `process\_tdt\_folder`/`load\_dataset`
accept:

* **`ols`** *(default)* — plain least-squares, no robustness at all. The
fit line can get dragged toward a severe artifact, but that's a
predictable weakness; unlike RANSAC/Huber below it never actively
discards or downweights a sample, so it can't mistake a real, large
biological transient for an artifact and distort the correction there.
* **`ransac`** — fits to random small subsets, keeps whichever
gets the most points within a residual threshold, refits on just those
inliers. Excludes artifact points from the fit entirely rather than
downweighting them — worth trying when artifacts are occasional and
severe enough (a fiber-cord twist, a brief motion burst) that this
trade-off clearly favors excluding them.
* **`huber`** — every point stays in the fit, but points beyond a
residual threshold get down-weighted instead of excluded. Gentler than
RANSAC's hard cut; better suited to noise elevated throughout the
recording rather than concentrated in a few bad stretches.

### `validate\_tdt\_folder(path) -> (bool, str)`

Checks whether a directory contains a valid TDT recording block. Returns
`(True, folder\_name)` if valid, `(False, error\_message)` if not.

### `get\_tdt\_struct(path)`

Loads the raw TDT block (streams, epocs, scalars) via the `tdt` SDK.
Handles a real SDK bug around epoc stores with mismatched onset/offset
counts by reading each epoc store with its own fresh call rather than
all at once.

### `get\_plot\_data(data, store\_name, channel=0, max\_points=None) -> (time, signal, fs)`

Extracts one stream's time series from a loaded TDT struct, with
optional downsampling.

### `correct\_bleaching(y, fs) -> (corrected, trend)`

Estimates and removes the photobleaching trend via masked curve fitting.

### `denoise\_signal(signal, fs, cutoff=5, order=2)`

Low-pass Butterworth filter for ΔF/F signals.

### `get\_event\_markers(data) -> list\[dict]`

Extracts behavioral event markers from every populated TDT epoc store
(auto-discovered — nothing about store names is hardcoded). Each onset
and offset becomes its own marker (`phase`: `"high"`/`"low"`), except the
`Note` store (instantaneous, text-labeled, no offset) and `Tick` (TDT's
own 1-second heartbeat, skipped entirely — not a real event).

**Returns** list of `{"time", "label", "color", "store", "phase"}`
(`phase` omitted for `Note` markers).

### `debounce\_events(times, min\_isi) -> list\[float]`

Collapses switch-bounce/double-tap duplicate timestamps — drops any
event within `min\_isi` seconds of the last *kept* event. Independent of
any fixed schedule; a `min\_isi` below the subject's real max event rate
leaves genuinely fast consecutive events intact.

\---

## Signal Analysis

### General helpers

#### `estimate\_sample\_rate(time\_array) -> float`

Sample rate (Hz) from a timestamp array, via the **median** inter-sample
interval — robust to one irregular gap skewing a naive
`1/(t\[1]-t\[0])` estimate.

#### `mean\_channels(arr) -> np.ndarray`

Collapses a multi-channel `(n\_channels, n\_samples)` array to its
across-channel mean (e.g. Oxysoft's per-detector o2hb/hhb/thb). A no-op
if `arr` is already 1D.

#### `compute\_group\_stats(trial\_matrix) -> (mean\_trace, sem\_trace)`

Mean + SEM across trials (rows) of a trial × time matrix. `sem\_trace` is
all zeros for fewer than 2 trials (SEM with ddof=1 is undefined below
that); `mean\_trace` is all zeros for 0 trials.

#### `compute\_marker\_intervals(markers) -> list\[dict]`

Time-since-previous-marker, two ways, for a list of markers from any
source (TDT epocs, Oxysoft events, manual). Each marker needs at least
`time`/`label` keys (`store` falls back to `label`, `phase` optional).

**Returns** one dict per marker: `{"time", "store", "label", "phase", "dt\_store", "dt\_global"}` — `dt\_store` is the interval since that
store's last event (for a high/low pair, a low marker's `dt\_store` *is*
its on-duration); `dt\_global` is the interval since whatever marker came
immediately before it, regardless of store. Both are `None` for the
first event in their sequence.

#### `smooth\_signal(data, fs, window\_sec=0.5)`

Moving-average smoothing filter.

#### `bin\_for\_heatmap(z\_seg, num\_bins=300)`

Bins a signal into `num\_bins` equal segments, for heatmap plotting.

### Z-score / event-triggered analysis

#### `get\_zscore\_slice(time\_array, signal, center\_t, window=None, pre=None, post=None) -> (time\_segment, z\_scored\_signal)`

Extracts and z-scores a window around one event time. Accepts either a
symmetric `window` (split evenly) or an asymmetric `pre`/`post` pair
(`pre`/`post` take precedence if both are given).

#### `compute\_event\_zscore\_peth(time\_array, signal, event\_times, pre, post, num\_bins=300) -> dict`

Z-scores and aligns **every occurrence** of one event type into a
trial × time matrix (GuPPy-style), for a stacked-heatmap + trial-average
view — as opposed to a single click-triggered PETH. Each trial is
z-scored independently against its own pre-event baseline, then
resampled onto a shared `num\_bins`-point relative-time axis so trials
with slightly different raw sample counts can still stack into one
matrix.

**Returns** `dict`:

|key|meaning|
|-|-|
|`time\_axis`|`(num\_bins,)`, relative time from `-pre` to `+post`|
|`trial\_matrix`|`(n\_valid\_trials, num\_bins)`|
|`trial\_event\_times`|the `event\_times` that produced a usable trial (too-close-to-recording-edge events are skipped)|
|`mean\_trace` / `sem\_trace`|`(num\_bins,)` each|

### FFT

#### `compute\_fft\_slice(time\_array, signal, center\_t, fs, window=None, pre=None, post=None) -> (freqs, power, seg\_x, seg\_y)`

Extracts a window around `center\_t` and computes its FFT, after mean
removal + linear detrending (eliminates the DC spike/slow drift so
physiological frequencies — breathing \~0.3Hz, heart rate \~1Hz — are
visible). Same symmetric-`window`-vs-asymmetric-`pre`/`post` convention
as `get\_zscore\_slice`.

#### `find\_fft\_peaks(freqs, power, n\_peaks=3) -> list\[dict]`

Top-N peaks in a power spectrum, as data. Returns
`\[{"freq\_hz", "power", "bpm"}, ...]` sorted by power descending — empty
if fewer than 3 usable frequency bins (≥0.05Hz) exist or nothing clears
the prominence bar.

#### `annotate\_fft\_peaks(ax\_f, freqs, power, color, n\_peaks=3)`

Same peak-finding as `find\_fft\_peaks`, but draws frequency/BPM labels
directly onto a matplotlib `Axes`.

### Curve fitting

#### `compute\_slope\_segment(x\_data, y\_data, p1\_idx, p2\_idx, padding\_pct=0.05) -> dict`

Least-squares linear regression slope between two index boundaries.
Returns `{"slope", "intercept", "crop\_x", "crop\_y", "x1", "y1", "x2", "y2"}`.

#### `fit\_model\_to\_segment(x\_seg, y\_seg, model\_fn, p0\_fn) -> dict`

Fits `model\_fn` (signature `f(x, \*params) -> y`, see
[Curve-fit models](#curve-fit-models)) to a segment via `scipy.optimize.curve\_fit`,
with `p0\_fn(x\_seg, y\_seg) -> initial\_guesses` supplying the starting
parameters. Returns `{"popt", "y\_fit", "r2", "success", "error"}`.

### Peak detection

#### `find\_significant\_peaks(time\_array, signal, z\_threshold=2.5, min\_distance\_sec=1.0, include\_troughs=False) -> list\[dict]`

Auto-detects statistically significant transients directly from the
signal — instead of trusting that externally-supplied event markers
actually line up with real activity. Z-scores the whole recording
against its own global mean/std (not per-event), then `scipy.signal.find\_peaks`
picks local maxima at or above `z\_threshold`, at least `min\_distance\_sec`
apart. `include\_troughs` also detects significant negative deflections
(off by default).

**Returns** `\[{"time", "z\_score", "kind": "peak"|"trough"}, ...]`, sorted
by time.

#### `find\_peak\_near\_events(time\_array, signal, event\_times, pre, post, z\_threshold=2.5, include\_troughs=False) -> list\[dict]`

Checks whether a significant peak actually shows up near each given
event time (baselined the same way as `get\_zscore\_slice`, i.e. relative
to that event's own local baseline — not the whole recording). Works for
one event or many occurrences of the same type.

**Returns** one dict per `event\_time` (same order):
`{"event\_time", "found", "peak\_time", "latency", "z\_score", "kind"}` —
`found` is `False` (with the rest `None`) if the window was unusable or
nothing cleared `z\_threshold`.

### Area under the curve

#### `compute\_auc\_from\_trace(x, y) -> dict`

Trapezoidal AUC, split by sign — the shared core `compute\_auc\_window`
and `compute\_auc\_matrix` both delegate to. `positive\_auc` integrates
only where `y >= 0`; `negative\_auc` integrates only where `y <= 0` and
is itself signed (≤ 0, not an absolute value); `total\_auc` is their sum
(equivalent to the plain net signed integral).

**Returns** `{"total\_auc", "positive\_auc", "negative\_auc"}` — all zero
if fewer than 2 points.

#### `compute\_auc\_window(time\_array, signal, center\_t, pre, post) -> dict`

Slices a pre/post window around an event and integrates it (same
windowing convention as `get\_zscore\_slice`). Deliberately does **not**
z-score first — AUC quantifies the signal in its own (already
dF/F-normalized) units. Returns `{"seg\_x", "seg\_y", "total\_auc", "positive\_auc", "negative\_auc"}`.

#### `compute\_auc\_matrix(time\_axis, trial\_matrix) -> dict`

Per-trial + group AUC for a trial × time matrix (see
`compute\_event\_zscore\_peth`). Returns `{"per\_trial": \[...], "group": {...}}`
— `per\_trial` one entry per row (same order), `group` computed on the
matrix's mean trace.

### Peri-event statistics

#### `compute\_peri\_event\_from\_trace(rel\_x, y, post, bin\_width=1.0) -> dict`

Peak amplitude, latency (time-to-peak), and mean-time-bins from an
already event-relative trace (`rel\_x`: 0 = event time). Peak/latency are
post-event only (`rel\_x >= 0` — a pre-event sample can't be a response),
using signed `argmax(|y|)` so a strong inhibitory (negative-going)
response is captured correctly, not just the largest positive value.

**Returns** `{"peak\_amplitude", "latency", "mean\_bins"}` — `mean\_bins` is
a list of `{"bin\_start", "bin\_end", "mean"}` covering `\[0, post]` in
`bin\_width`-second chunks (last bin truncated, not dropped, if `post`
isn't evenly divisible).

#### `compute\_peri\_event\_matrix(time\_axis, trial\_matrix, post, bin\_width=1.0) -> dict`

Per-trial + group version of the above, mirroring `compute\_auc\_matrix`.
Returns `{"per\_trial": \[...], "group": {...}}`, each entry
`{"peak\_amplitude", "latency", "mean\_bins"}`.

\---

## Non-destructive splicing

Both functions work on 1D arrays *and* 2D `(n\_channels, n\_samples)`
arrays via `extra\_channels` (sliced along the last axis) — so Oxysoft's
multi-detector data splices exactly like TDT's single-channel data, no
special-casing needed by the caller.

### `splice\_keep\_inside(x, raw, corr, markers, detected\_markers, start, end, extra\_channels=None) -> dict | None`

Trims `x`/`raw`/`corr` to `\[start, end]`, filters both marker lists to
the same range. `extra\_channels` (optional): any other same-length
arrays that need trimming identically — e.g. TDT's raw per-wavelength
channels, or Oxysoft's `(n\_detectors, n\_samples)` arrays.

**Returns** `{"x", "raw", "corr", "markers", "detected\_markers", "n\_samples", "extra\_channels"}` (copies, not views), or `None` if the
range doesn't contain at least 2 samples. Marker timestamps stay
absolute (not re-zeroed).

### `splice\_cut\_out(x, raw, corr, markers, detected\_markers, start, end, extra\_channels=None) -> dict | None`

Removes `\[start, end]` and stitches the remainder together, shifting
everything after the cut backward by the cut's duration so the timeline
stays contiguous. Markers inside the cut are dropped; markers after it
are shifted by the same amount. Same `extra\_channels` support as above.

**Returns** same shape as `splice\_keep\_inside`, or `None` if there isn't
usable signal on both sides of the cut to stitch together.

\---

## Curve-fit models

All follow `f(x, \*params) -> y`, ready to pass straight into
`fit\_model\_to\_segment`.

|function|formula|
|-|-|
|`linear\_model(x, m, b)`|`y = m\*x + b`|
|`single\_exponential\_model(x, a, b, c)`|`y = a\*exp(-b\*x) + c`|
|`double\_exponential\_model(x, a, b, c, d, k)`|`y = a\*exp(-b\*x) + c\*exp(-d\*x) + k` — physical bleaching model|
|`exponential\_rise\_model(x, a, b, c)`|`y = a\*(1 - exp(-b\*x)) + c` — e.g. venous occlusion|
|`gaussian\_model(x, a, mu, sigma)`|`y = a\*exp(-(x-mu)^2 / (2\*sigma^2))`|
|`sinusoidal\_model(x, a, f, phi, c)`|`y = a\*sin(2\*pi\*f\*x + phi) + c`|
|`visibility\_model(beta, a, v, beta\_c, period)`|`y = (a/2) \* (1 - v\*sin((beta - beta\_c) / period))` — photon-entanglement visibility|

\---

## Text Field Study

A pipeline for comparing free-text survey/response fields across
subjects via sentence-embedding similarity — e.g. "did this subject's
answer to field A resemble their answer to field B more than chance
pairing would predict?" Needs the optional `sentence-transformers`
package (see [Optional dependencies](#optional-dependencies)).

### `run\_field\_study\_pipeline(folder\_path, text\_fields, delta\_pair=None, paired\_fields=None, model\_name="all-MiniLM-L6-v2", n\_null=200, rng\_seed=None, file\_glob="P-\*.json", min\_words=5) -> pandas.DataFrame`

The single entry point — runs the whole pipeline (load → flag
low-quality → embed → optional delta vector → optional paired
similarity + permutation test + word-count confound check) and folds
everything into one DataFrame, one row per subject.

* `delta\_pair` (`(field\_from, field\_to)`, optional): adds a
`delta\_magnitude` column — `‖vec(field\_to) - vec(field\_from)‖`.
* `paired\_fields` (`\[(field\_a, field\_b, pair\_name), ...]`, optional):
adds per pair `sim\_<name>`, `null\_mean\_<name>`, `null\_std\_<name>`,
`pvalue\_<name>`/`effect\_size\_<name>` (permutation test), and
`wc\_confound\_r\_<name>`/`wc\_confound\_p\_<name>` (word-count confound
check). The test-level values repeat identically across every row for
that pair — one number per pair, not per subject.

The stage functions below are also available individually for anyone
who wants an intermediate result.

### `load\_field\_study\_folder(folder\_path, text\_fields, file\_glob="P-\*.json") -> pandas.DataFrame`

Loads every file matching `file\_glob` into one row-per-subject
DataFrame, adding a `wordcount\_<field>` column per text field.

### `peek\_fields(folder\_path, file\_glob="P-\*.json") -> list\[str]`

Field names (dict keys) found in the *first* matching file, without
loading the whole folder — for a GUI to show real field names to pick
from rather than requiring them typed blind.

### `flag\_low\_quality(df, text\_fields, min\_words=5) -> pandas.DataFrame`

Flags (doesn't drop) near-empty responses. Adds `low\_quality\_<field>`
per field and `any\_low\_quality` (OR across fields). Must run after
`load\_field\_study\_folder` (needs the `wordcount\_<field>` columns).

### `embed\_text\_fields(df, fields, model\_name="all-MiniLM-L6-v2") -> dict`

Embeds each field with a `sentence-transformers` model, L2-normalized
(so a plain dot product is cosine similarity). Returns
`{field: (n\_subjects, dim) ndarray, ...}`. Raises a clear `ImportError`
with an install hint if `sentence-transformers` isn't installed.

### `compute\_delta\_vector(embeddings, field\_from, field\_to) -> (delta\_vectors, delta\_magnitude)`

Per-subject `vec(field\_to) - vec(field\_from)`, plus its magnitude.

### `compute\_paired\_similarity(embeddings, paired\_fields, n\_null=200, rng\_seed=None) -> dict`

Cosine similarity between each field pair for the same subject, plus a
null distribution built by repeatedly shuffling one field's vectors
across subjects. Returns, per pair name:

```python
{
    'same\_subject\_similarity': ndarray,   # (n\_subjects,)
    'null\_mean': float, 'null\_std': float,
    'null\_values': ndarray,               # (n\_null \* n\_subjects,), pooled
    'null\_shuffle\_means': ndarray,        # (n\_null,) — the actual null
                                           # distribution for testing the MEAN
}
```

### `permutation\_test\_similarity(same\_subject\_similarity, null\_shuffle\_means) -> dict`

One-tailed permutation test: is observed mean same-subject similarity
higher than chance pairing? Returns `{"observed\_mean\_similarity", "p\_value", "effect\_size", "null\_shuffle\_mean", "null\_shuffle\_std"}`.

### `wordcount\_confound\_check(df, paired\_fields) -> dict`

Pearson correlation between each pair's combined word count and its
similarity, across subjects — a significant positive correlation would
mean "longer answers just look more similar," undercutting a similarity
claim. Returns `{pair\_name: {"r", "p\_value"}, ...}` (both `NaN` if fewer
than 2 subjects or word count has no variance).

\---

## Text Field Study — Statistical Validation

A companion module that stress-tests a Text Field Study result: is the
similarity real, or could it be a word-count artifact, a single outlier
subject, or noise that wouldn't survive multiple-comparisons correction?

### `run\_validation\_pipeline(folder\_path, text\_fields, paired\_fields, model\_name="all-MiniLM-L6-v2", n\_null=200, n\_boot=1000, rng\_seed=None, file\_glob="P-\*.json", id\_field="participant\_id") -> pandas.DataFrame`

Full validation pipeline: load → embed → paired similarity → build the
summary table below. Self-contained (recomputes embeddings rather than
reusing a prior `run\_field\_study\_pipeline` call). `paired\_fields` is
required here — there's nothing to validate without at least one pair.

### `build\_validation\_summary(df, paired\_fields, similarity\_results, id\_field="participant\_id", n\_boot=1000, rng\_seed=None) -> pandas.DataFrame`

One row per field pair:

|column|meaning|
|-|-|
|`pair`, `field\_a`, `field\_b`|which fields|
|`observed\_mean\_similarity`|mean same-subject similarity|
|`p\_value`|permutation-test p-value (one-tailed)|
|`p\_value\_fdr`|**use this one**, not raw `p\_value` — Benjamini-Hochberg corrected across all pairs in the table|
|`cohens\_d`|effect size vs. the pooled chance-pairing null|
|`wc\_coef\_a`/`wc\_pvalue\_a`, `wc\_coef\_b`/`wc\_pvalue\_b`|word-count regression coefficient/p-value per field, controlling for the other — significant means part of the effect may just be verbosity|
|`regression\_r\_squared`|variance in similarity explained by both word counts (low = reassuring)|
|`ci\_lower`, `ci\_upper`|bootstrap 95% CI on `observed\_mean\_similarity`|
|`n\_flagged\_loo`|subjects flagged by leave-one-out sensitivity (0 is reassuring)|
|`flagged\_participant\_ids`|which ones|

### `cohens\_d(group1, group2) -> float`

Standardized mean difference between two distributions (pooled std
units) — rough guide \~0.2 small, \~0.5 medium, \~0.8 large. Typically
called with `group1` = same-subject similarities, `group2` = pooled
null similarities.

### `benjamini\_hochberg(p\_values) -> ndarray`

False-discovery-rate correction across multiple hypothesis tests — less
conservative than Bonferroni while still controlling the *expected
proportion* of false positives among results called significant.
Returns corrected p-values ("q-values"), same order as input.

### `wordcount\_controlled\_regression(df, field\_a, field\_b, sim\_col) -> dict`

OLS regression: `similarity \~ wordcount\_<field\_a> + wordcount\_<field\_b>`.
Tests whether either field's length predicts similarity independent of
the other. **Requires `statsmodels`** (hard import, no fallback).
Returns `{"r\_squared", "coef\_const", "coef\_wordcount\_<field\_a>", "coef\_wordcount\_<field\_b>", "pvalue\_const", "pvalue\_wordcount\_<field\_a>", "pvalue\_wordcount\_<field\_b>", "model"}` (`model` is the full
`statsmodels` `RegressionResults` object).

### `bootstrap\_mean\_ci(values, n\_boot=1000, ci=0.95, rng\_seed=None) -> dict`

Bootstrap confidence interval on the mean (resample with replacement,
`n\_boot` times) — doesn't assume normality, unlike a textbook
`mean ± 1.96·SE` interval. Returns `{"mean", "ci\_lower", "ci\_upper", "boot\_means"}`.

### `leave\_one\_out\_sensitivity(values, ids=None) -> dict`

Recomputes the mean with each subject dropped one at a time; flags any
subject whose removal shifts the mean by more than one standard
deviation of the resulting leave-one-out-means distribution. Catches a
single subject quietly dominating the result. A flagged subject isn't
automatically wrong/excludable — it's worth a manual look.

**Returns** `{"full\_mean", "loo\_means", "shifts", "threshold", "flagged\_ids"}`.

\---

## Optional dependencies

`numpy`, `scipy`, `pandas`, `tdt`, and `scikit-learn` are regular,
required dependencies (installed automatically by
`pip install ZaksPhysicsLibrary`) — `scikit-learn` in particular backs
*all* of `process\_tdt\_folder`'s regression methods including `"ols"`,
since `processing\_TDT.py` imports it unconditionally at module level,
not just when RANSAC/Huber are actually selected.

The two below are genuinely soft — each has a guarded import that fails
with a clear, actionable message instead of crashing obscurely, rather
than being unavailable outright:

|package|needed for|behavior if missing|
|-|-|-|
|`openpyxl`|`load\_any\_file` on `.xlsx`/`.xls`|`ImportError` at call time: "Install openpyxl to read .xlsx files: pip install openpyxl"|
|`sentence-transformers`|`embed\_text\_fields` (and everything downstream: `compute\_delta\_vector`, `compute\_paired\_similarity`, `permutation\_test\_similarity`, `wordcount\_confound\_check`, `run\_field\_study\_pipeline`, `run\_validation\_pipeline`, `build\_validation\_summary`)|`ImportError` with an install hint (`pip install sentence-transformers`)|

`sentence-transformers` pulls in `torch`/`transformers` (hundreds of MB
to 1GB+) — it's actually listed as a normal, unconditional dependency in
`pyproject.toml` despite being guarded at the code level like a soft
one; if you're building a lightweight install and don't need Text Field
Study, it's worth installing everything else by hand rather than via
`pip install ZaksPhysicsLibrary` directly.

`statsmodels` is also a required `pyproject.toml` dependency, but
`wordcount\_controlled\_regression` is the *only* function that actually
imports it (as a hard, unguarded `import` inside the function body) — so
while it's always installed alongside the package, it's only ever
exercised if you call that one function.

