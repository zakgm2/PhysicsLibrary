# Changelog — PhysicsLibrary

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
