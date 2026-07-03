# Changelog — PhysicsLibrary

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
