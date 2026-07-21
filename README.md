# PhysicsLibrary

Data processing and analysis library backing [PhysicsAnalysis](https://github.com/zakgm2/PhysicsAnalysis) — file parsing, signal processing, and curve-fitting logic, with no GUI code of its own. Any interface (tkinter, PyQt6, a script, a notebook) can sit on top of it.

---

## What it does

- **Loads lab data** from three instrument formats plus generic tabular files:
  - **TDT** (Tucker-Davis Technologies) fibre photometry tanks
  - **Oxysoft / Artinis** (Oxymon, OctaMon, PortaMon …) NIRS `.txt` exports
  - **Terranova Prospa** `.pt2` EFNMR/MRI 2D images
  - Generic **Excel / CSV / TSV / plain text**, with automatic sub-table detection for side-by-side data layouts on one sheet
- **Processes signals** — bleach correction, denoising, Z-score PETH slicing, FFT with peak annotation, slope/segment analysis
- **Fits curves** — linear, single/double exponential, exponential rise, Gaussian, sinusoidal, and a photon-entanglement visibility model, all via `scipy.optimize.curve_fit`
- **Analyses text field studies** — one JSON file per subject with several free-text fields; pick any pair of fields to compare directly (e.g. does the answer to one question track another for the same subject). Word counts, data-quality flagging, sentence-transformers embeddings, an optional delta-vector magnitude between two fields, and an optional paired-similarity metric per pair with a permutation test and a word-count confound check. Domain-agnostic — field names and which pairs to compare are supplied by the caller, nothing is hardcoded to one study
- **Validates the similarity metric statistically** — Benjamini-Hochberg FDR-corrected permutation-test p-values, Cohen's d effect size, a word-count-controlled OLS regression (statsmodels), a bootstrap confidence interval on the mean, and a leave-one-out sensitivity check, one row per field pair, with docstrings explaining what each statistic means

---

## Structure

```
PhysicsLibrary/
  __init__.py            Public API — see below
  dataset.py              Dataset struct, DataFormat enum, format detection, folder picker
  file_parser.py           Top-level dispatcher: load_dataset(), load_dataset_file()
  file_parser_generic.py    Generic Excel/CSV/TSV/text parser with sub-table detection
  processing_TDT.py          TDT tank reading, bleach correction, denoising, event markers
  analysis.py                 PETH/Z-score, FFT, slope segments, curve-fit runner
  models.py                    Parametric model functions for curve fitting
  text_field_study.py           Grouped-text-field study pipeline (embeddings, delta vector, paired similarity)
  field_study_validation.py      Statistical validation for the paired-similarity metric (permutation test, Cohen's d, regression, bootstrap CI, leave-one-out)
  loaders/
    tdt_loader.py               Wraps processing_TDT into a Dataset
    oxysoft_loader.py            Oxysoft .txt parsing (folder + single-file) into a Dataset
    pt2_loader.py                 .pt2 EFNMR/MRI image parser
```

Each loader/parser is single-purpose and has no knowledge of the others — `file_parser.py` is the only place that ties format detection to the right loader.

---

## Installation

```bash
pip install git+https://github.com/zakgm2/PhysicsLibrary.git
```

Or as a dependency in another project's `requirements.txt`:

```
git+https://github.com/zakgm2/PhysicsLibrary.git
```

### Requirements

- Python 3.10+
- `numpy`, `scipy`, `tdt`, `pandas`, `sentence-transformers`, `statsmodels` (installed automatically)
- `sentence-transformers` pulls in `torch`/`transformers` as transitive dependencies — a genuinely heavy install (hundreds of MB) if you only need the signal-processing side; only actually loaded when you call `embed_text_fields`/`run_field_study_pipeline`
- `openpyxl` — only needed for `.xlsx`/`.xls` files; imported lazily with a clear error if missing when you actually try to load Excel

---

## Usage

```python
import PhysicsLibrary as pl

# Detect + load a TDT tank or Oxysoft export folder
fmt     = pl.detect_format(folder_path)
dataset = pl.load_dataset(folder_path, fmt)

# Or load a single Oxysoft .txt file directly
dataset = pl.load_dataset_file(file_path)

# Every loader returns the same universal Dataset struct
dataset.source_format   # "TDT" | "Oxysoft"
dataset.sample_rate      # Hz
dataset.signals            # (num_channels, num_samples)
dataset.channel_names        # list[str]
dataset.events                # [{'label': str, 'sample': int}, ...]
```

```python
# Generic tabular data (Excel/CSV/TSV/text) — returns one GenericTable per
# detected sub-table, since a single sheet can contain several side-by-side
tables = pl.load_any_file(path)
table  = tables[0]
table.headers   # list[str]
table.data      # (n_rows, n_cols) float64, NaN for missing

# Terranova .pt2 EFNMR/MRI image — returns a raw 2D array, not a Dataset
img = pl.load_pt2(path)   # (n, n) float32
```

```python
# Analysis
x_seg, z = pl.get_zscore_slice(time_array, signal, center_t, window=30)
freqs, power, seg_x, seg_y = pl.compute_fft_slice(time_array, signal, center_t, fs)
pl.annotate_fft_peaks(ax, freqs, power, color='blue')   # matplotlib peak labels

# Curve fitting
result = pl.fit_model_to_segment(x_seg, y_seg, pl.single_exponential_model, p0_fn)
result["popt"], result["r2"], result["y_fit"]
```

```python
# Text field study — one JSON file per subject, e.g. P-0001.json. Pick
# pairs of fields to compare directly; no grouping concept needed.
fields = pl.peek_fields(folder_path)              # see what fields exist before picking pairs
df = pl.run_field_study_pipeline(
    folder_path,
    text_fields=["q1", "q2", "q3", "q4"],
    delta_pair=("q1", "q2"),                        # optional: how much did q2 change from q1
    paired_fields=[("q1", "q3", "pair1")],           # optional: does q1 track q3
)
# df has one row per subject: wordcount_<field>, low_quality_<field>, delta_magnitude,
# sim_<pair>, pvalue_<pair>, effect_size_<pair>, wc_confound_r_<pair>, ...
```

```python
# Statistical validation of the paired-similarity metric — one row per pair
summary = pl.run_validation_pipeline(
    folder_path,
    text_fields=["q1", "q3"],
    paired_fields=[("q1", "q3", "pair1")],
)
# summary: p_value, p_value_fdr, cohens_d, wc_coef_a/b + wc_pvalue_a/b,
# regression_r_squared, ci_lower/ci_upper, n_flagged_loo, flagged_participant_ids
```

See [FIELD_STUDY_METHODOLOGY.md](FIELD_STUDY_METHODOLOGY.md) for why each statistic in the
validation step is a sound, standard technique — useful if anyone asks.

See [CHANGELOG.md](CHANGELOG.md) for the version history.

---

## Public API

Everything importable from `PhysicsLibrary` directly:

| Category | Names |
|----------|-------|
| Format detection | `detect_format`, `detect_format_file`, `DataFormat`, `Dataset` |
| Loading | `load_dataset`, `load_dataset_file`, `load_any_file`, `load_pt2` |
| TDT processing | `process_tdt_folder`, `validate_tdt_folder`, `get_tdt_struct`, `get_plot_data`, `correct_bleaching`, `denoise_signal`, `get_event_markers`, `debounce_events` |
| Analysis | `get_zscore_slice`, `smooth_signal`, `bin_for_heatmap`, `compute_fft_slice`, `annotate_fft_peaks`, `compute_slope_segment`, `fit_model_to_segment` |
| Curve fit models | `linear_model`, `single_exponential_model`, `exponential_rise_model`, `double_exponential_model`, `gaussian_model`, `sinusoidal_model`, `visibility_model` |
| Text field study | `run_field_study_pipeline`, `load_field_study_folder`, `peek_fields`, `flag_low_quality`, `embed_text_fields`, `compute_delta_vector`, `compute_paired_similarity`, `permutation_test_similarity`, `wordcount_confound_check` |
| Field study validation | `run_validation_pipeline`, `build_validation_summary`, `cohens_d`, `benjamini_hochberg`, `wordcount_controlled_regression`, `bootstrap_mean_ci`, `leave_one_out_sensitivity` |
