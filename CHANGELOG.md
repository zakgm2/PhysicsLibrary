# Changelog — PhysicsLibrary

---

## Version 1.10.0:
  New:
  - process_tdt_folder() now returns "motion_correction_inlier_fraction": the fraction (0-1)
    of samples RANSAC kept as inliers when fitting the 415-vs-465 motion correction line, or
    None if there was no isosbestic reference stream to correct against. Lets a caller (e.g. the
    GUI) show how much of the recording was judged to be motion artifact and excluded from the fit.

  Changed:
  - get_event_markers() now skips the 'Tick' epoc store entirely. It's TDT's own 1-second
    heartbeat/reference signal, not a behavioral event — every recording has it, nothing ever
    wants to plot/analyze/splice against it, and its regularity would otherwise flood every
    marker and event-name picker a caller builds from this function's output.

  Fixed:
  - _robust_linear_fit()'s noise estimate (used to set RANSAC's residual_threshold) was based on
    consecutive-sample differences, which measures noise at the timescale of one sample —
    fine in principle, but at TDT's actual sampling rates (~1 kHz, adjacent samples ~1ms apart)
    that's dominated by ADC/thermal noise far smaller than the residual scale a real 415-vs-465
    regression produces once genuine biological signal is accounted for. Confirmed on a real
    recording (PFC-GCaMP-Sucrose): diff-based noise estimation set a threshold ~350x tighter
    than the fit's actual residual scale (OLS residual std ≈ 5.19 vs. a 0.0146 threshold),
    rejecting 98% of samples as "outliers" — including essentially all real signal. Replaced
    with the MAD of an initial OLS fit's own residuals, which is on the right scale by
    construction and stays robust to the artifacts it's meant to exclude; same recording now
    lands at ~95% inliers. (1.9.1's sqrt(2)/duplicate-scaling fix and this session's earlier
    zero-diff guard were both correct fixes to the diff-based approach, but the approach itself
    was the wrong tool at this sampling rate — this replaces it rather than patching further.)

---

## Version 1.9.1:
  Fixed:
  - processing_TDT.py: _robust_linear_fit()'s noise estimate (used to set RANSAC's
    residual_threshold — see 1.9.0 below) applied the MAD-to-standard-deviation scale factor
    twice (`/ 0.6745 * 1.4826`, where 1.4826 ≈ 1/0.6745), inflating the estimate by roughly 2x,
    and never corrected for consecutive-sample differencing doubling the variance
    (Var(y[i+1]-y[i]) = 2·Var(y) for independent samples) — so the residual threshold RANSAC
    used was calibrated well above the recording's actual noise level, meaning it excluded fewer
    genuine outliers than intended. Corrected to `MAD / (0.6745 * sqrt(2))` — each factor applied
    exactly once. Verified against synthetic data with known noise: the old formula estimated
    noise at roughly 2x the true level; the corrected one lands within ~2%.

---

## Version 1.9.0:
  Changed:
  - processing_TDT.py: process_tdt_folder() now motion-corrects (regresses the isosbestic/415
    stream onto the signal/465 stream) with RANSAC robust regression (sklearn's
    RANSACRegressor) instead of ordinary least squares (np.polyfit). OLS lets exactly the kind
    of thing this regression exists to remove — a burst of motion artifact, a fiber-cord twist —
    drag the fitted line toward itself, corrupting the "motion-free" prediction across the whole
    recording rather than just where the artifact happened; RANSAC instead fits to random small
    subsets, keeps whichever gets the most points within a residual threshold ("inliers"), and
    does one final fit on just those — points that never look like they belong to the same line
    are excluded from the fit entirely, not merely downweighted. The residual threshold is set
    explicitly (3x a robust noise estimate from consecutive-sample differences) since sklearn's
    own default is calibrated for roughly-flat data and comes out far too loose here (y's spread
    is dominated by the real 415-vs-465 trend, not noise) — left at the default, RANSAC would
    accept every point as an inlier and silently degrade to a plain OLS fit. Verified against
    synthetic data with an injected outlier burst: recovers the true slope/intercept almost
    exactly where OLS was measurably pulled off, and against a real recording with a
    poorly-conditioned 415/465 relationship (OLS previously threw `RankWarning: Polyfit may be
    poorly conditioned`) where the bleaching-trend baseline that depends on this fit went from
    swinging across 6 orders of magnitude (0.5 to 117,177 — a failed fit) to a tight, sane 0.49–0.88
    range. New dependency: scikit-learn (was already present transitively via
    sentence-transformers, now declared explicitly). This changes dF/F output values for every
    TDT recording that has an isosbestic reference stream — not a bug fix to a wrong number, but
    a real change in which regression method computes it.
    
  New:
  - processing_TDT.py: process_tdt_folder() now also returns "channels" — a list of
    {"key", "label", "y"} for each raw per-wavelength stream found (always "main_driver",
    plus "isosbestic" too if a 415 reference stream exists), the un-motion-corrected,
    un-normalized traces. Lets a caller plot/analyze the isosbestic control or the main
    driver/probe channel directly instead of only the final "raw"/"corr" result. Nothing about
    stream naming or count is hardcoded beyond the existing 465/415 detection, so a recording
    with no isosbestic stream simply has no "isosbestic" entry — still works.
  - splice.py: splice_keep_inside()/splice_cut_out() take an optional extra_channels dict of
    {name: array} — any other arrays sample-aligned with x (e.g. the new raw per-wavelength
    channels above) that need trimming/cutting identically to x/raw/corr, returned back under
    "extra_channels". Optional and backward compatible: omitting it behaves exactly as before.

---

## Version 1.8.0:
  New:
  - processing_TDT.py: debounce_events(times, min_isi) — collapses switch-bounce/double-tap
    duplicate event timestamps (e.g. a lever registering more than one contact for what was
    physically a single press) by sorting the timestamps and dropping anything within min_isi
    seconds of the previous *kept* event. Independent of any fixed-ratio schedule (FR1, FR3, ...)
    — it works on raw inter-event spacing, not an expected press count, and a min_isi comfortably
    below the animal's real max press rate leaves genuinely fast consecutive presses (e.g. an FR3
    burst) intact while still removing sub-threshold hardware bounce.

---

## Version 1.7.0:
  New:
  - splice.py: splice_keep_inside(x, raw, corr, markers, detected_markers, start, end) and
    splice_cut_out(x, raw, corr, markers, detected_markers, start, end) — non-destructive
    time-range edits of a signal (trim to a range, or remove one and stitch the remainder back
    together). cut_out shifts everything after the removed range backward by its duration so the
    timeline stays contiguous — a gap in x would break any downstream analysis assuming uniform
    sampling right at the cut boundary — and drops/shifts marker dicts to match. Pure array/dict
    operations, no knowledge of any GUI's state shape.

  Fixed:
  - loaders/oxysoft_loader.py: load_oxysoft_file() failed with a cryptic
    "not enough values to unpack (expected 2, got 1)" when a file's Legend block didn't match the
    expected O2Hb/HHb column format (o2hb ended up shape (0,), a 1-tuple). Now raises a clear
    ValueError naming the actual problem.

---

## Version 1.6.0:
  New:
  - analysis.py: compute_event_zscore_peth(time_array, signal, event_times, pre, post, num_bins) —
    Z-scores every occurrence of one event type against its own pre-event baseline and aligns
    them into a trial x time matrix (plus mean/SEM traces), for a GuPPy-style stacked-heatmap
    PETH rather than a single click-triggered one.
  - analysis.py: find_significant_peaks(time_array, signal, z_threshold, min_distance_sec,
    include_troughs) — z-scores the whole recording against its own global mean/std and returns
    local peaks (scipy.signal.find_peaks) at or above threshold, for finding candidate events
    directly from the signal rather than trusting externally-supplied event markers.
  - analysis.py: find_peak_near_events(time_array, signal, event_times, pre, post, z_threshold,
    include_troughs) — checks whether a statistically significant peak actually shows up near
    each given event time (single event or many occurrences of one type), reporting found/
    latency/z-score per event, so alignment between a marker and the real signal can be verified
    rather than assumed.

  Fixed:
  - processing_TDT.py: get_event_markers() — a level/buffered-logic epoc store already "high"
    the instant recording started got a spurious onset marker at exactly t=0 (TDT's synthetic
    starting-state entry, since it has no signal history before t=0 — not a real event). Now
    filtered out, mirroring the existing offset == inf guard for the opposite edge case (a
    strobe still active at the recording's end).

---

## Version 1.5.0:
  New:
  - text_field_study.py: a domain-agnostic pipeline for a study where each subject produces
    one JSON file with several free-text fields, letting a caller pick any pair of fields to
    compare directly (e.g. does the answer to one question track another for the same
    subject) — nothing about which fields exist or which pairs to compare is hardcoded.
    run_field_study_pipeline() is the one-call entry point: loads every file matching a glob
    pattern in a folder into a DataFrame, flags (never drops) near-empty responses as a data
    quality signal, embeds only the fields actually referenced by a comparison or delta pair
    with a sentence-transformers model, optionally computes a delta-vector magnitude between
    two fields, and optionally computes per-pair cosine similarity for each subject plus a
    permutation test (p-value, effect size) of whether same-subject similarity beats a
    shuffled-pairing null, and a word-count-vs-similarity confound check. peek_fields() reads
    just the field names out of a folder's first file so a caller can show a user their
    actual fields before picking which to compare.
  - field_study_validation.py: statistical validation for the paired-similarity metric above.
    run_validation_pipeline() (or build_validation_summary() if you already have similarity
    results) produces one row per field pair: a permutation-test p-value (with
    Benjamini-Hochberg FDR correction across every pair tested together), Cohen's d effect
    size against the pooled null distribution, an OLS regression of similarity on both
    fields' word counts (statsmodels — checks whether the effect is just a "longer answers
    look more similar" artifact), a bootstrap 95% confidence interval on the mean similarity,
    and a leave-one-out sensitivity check flagging any subject whose removal shifts the mean
    by more than one standard deviation of the leave-one-out distribution. Every function's
    docstring explains what its statistic means, not just how to call it. Every one of these
    functions returns NaN rather than crashing when there isn't enough data for a statistic
    to be defined (e.g. a correlation needs at least 2 subjects) — a genuinely undefined
    result, not an error.
  - New dependencies: pandas, sentence-transformers, statsmodels.

---

## Version 1.4.0:
  New:
  - get_event_markers() now emits both edges of every epoc's state — onset ("high") and
    offset ("low") — tagged with a new 'phase' key, instead of only onset. A pump or light's
    on-duration needs both edges; something like a lever press usually only needs the
    onset, so a caller now gets to choose per store instead of the library deciding for it.
    The 'Note' store is unaffected — its markers are instantaneous free-text annotations,
    not a state with a duration, so they never get a 'phase' key. Offsets still equal to
    infinity (a strobe/epoch still active when the recording was stopped) are skipped, since
    there's no real timestamp there.

---

## Version 1.3.0:
  New:
  - get_event_markers() now extracts markers from every populated epoc store in a TDT
    recording, not just 'Note' — I/O strobes (PP1/PP2/EE1/EE2/Pmp/Tne/ShK), Epoch Event
    Storage stores (L1P/L2P/L1E/L2E/pump/Tne), and the Tick reference all become their own
    marker group. Each marker dict now includes a 'store' key so a caller (e.g. a GUI) can
    group/toggle markers by source store instead of showing every store at once. Nothing
    about which stores exist is hardcoded — this walks whatever data.epocs actually
    contains, so it works for any TDT block regardless of Synapse configuration.
  - get_zscore_slice() and compute_fft_slice() now accept optional pre/post parameters for
    an asymmetric window around an event (e.g. 10s before, 20s after), instead of only a
    symmetric total `window` split evenly. `window` is still supported and used as the
    default if pre/post aren't given. get_zscore_slice()'s baseline is now computed from
    the actual pre-event portion of the segment rather than "the first half," since those
    differ once pre != post.

  Bug fixes:
  - Fixed get_tdt_struct() crashing (IndexError from inside the tdt SDK) on recordings
    where any epoc store has a mismatched onset/offset count — e.g. a strobe/epoch that
    was still active when the recording was stopped, a completely normal way for a session
    to end. Root cause: the tdt SDK breaks when reading multiple epoc stores in one call if
    any of them has this mismatch. Fixed by reading each epoc store with its own
    tdt.read_block() call (which correctly reconstructs the missing offset) instead of
    batching them; any store that still fails on its own (a genuine data issue, not this
    bug) is skipped with a warning instead of crashing the whole load.

  Cleanup:
  - Removed dataset.choose_file() (and its tkinter dependency) — a GUI file-dialog concern
    that had no business in a data-processing library, and was unused by any caller (the
    Qt GUI opens its own QFileDialog directly). No replacement needed.

---

## Version 1.2.2:
  Packaging:
  - Fixed the PyPI package name in pyproject.toml (was "PhysicsLibrary", registered trusted
    publisher is "ZaksPhysicsLibrary") — this mismatch caused the GitHub Actions trusted
    publishing OIDC exchange to fail on upload. No code changes.

---

## Version 1.2.1:
  Refactor:
  - Moved load_pt2() from file_parser_generic.py into file_parser.py where it belongs
    alongside TDT and Oxysoft parsers. Public API unchanged.

---

## Version 1.2.0:
  New features:
  - load_pt2(): Terranova Prospa .pt2 EFNMR/MRI image parser added to file_parser_generic.py.
    Detects magnitude image data after the LAER marker, supports all square power-of-two sizes
    (16x16 through 256x256) with a perfect-square fallback.
  - Exported load_pt2 from the top-level PhysicsLibrary package.
---

## Version 1.1.0:
  New features:
  - Generic file parser (file_parser_generic.py): supports Excel (.xlsx), CSV, TSV, and plain
    text with automatic sub-table detection for side-by-side data layouts.
  - Open button consolidated into a single dropdown menu (Open TDT / Open TXT Oxysoft / Open Excel).
  - Excel/generic data now loads directly into the main GUI plot with full snap and hover support.
  - TSI Fit Factor extracted from Oxysoft .txt files and displayed as [FF: x.x%] in the legend.
  - Curve fit parameters can now be copied to clipboard or exported as a CSV file.
  - Grid toggle checkbox added to the toolbar.
  - Marker enhancements: font size and colour are now editable both when adding and via right-click
    edit. Colour options include green, red, blue, orange, purple, and black (matching TDT default).

  Bug fixes and performance:
  - Blit-based hover animation: tracker dots and connector line are drawn via restore_region /
    draw_artist / blit instead of canvas.draw_idle(), making hover ~20-50x faster.
  - Scroll zoom debounced (150 ms) and scale reduced to 1.1x per tick for smoother zooming.
  - Tracker dots no longer disappear during scroll zoom; dots no longer cause autoscale zoom.
  - Rect-select zoom no longer accidentally triggers a curve fit click on release.
  - Oxysoft hover snap now correctly targets mean lines (O2Hb, HHb, tHb) by linewidth filter.
  - Edit Attributes changes now persist correctly across all view interactions (zoom, pan, hover).
---

## Version 1.0.0: 
  - Fully loadable data sets, graphing functionality, so far only TDT and Oxysoft.

---

## Version 0.1.0: 
  - Made a GUI capable of handling different file types. Parser implemented.
