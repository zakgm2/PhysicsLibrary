"""
file_parser.py
--------------
Detects the recording format inside a selected folder and returns a
universal Dataset struct (dataclass).  Currently supports:

  - TDT  (Tucker-Davis Technologies) – detected by proprietary file extensions
  - Oxysoft / Artinis (Oxymon, OctaMon, PortaMon …) – detected by .txt export

Usage
-----
    from file_parser import choose_file, detect_format, load_dataset, DataFormat

    folder_path, folder_name = choose_file()
    fmt = detect_format(folder_path)
    dataset = load_dataset(folder_path, fmt)
"""

from __future__ import annotations  # must be first

import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from tkinter import filedialog

from typing import Optional
import tdt
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import numpy as np
from PhysicsLibrary import models, processing_TDT

# ---------------------------------------------------------------------------
# Folder selection
# ---------------------------------------------------------------------------

def choose_file(parent_window=None) -> tuple[Optional[str], Optional[str]]:
    """
    Opens a native folder selection dialog.

    Returns
    -------
    (folder_path, folder_name)  or  (None, None) if cancelled.
    """
    folder_path = filedialog.askdirectory(
        parent=parent_window,
        title="Open Lab Data Folder",
    )
    if not folder_path:
        return None, None
    return folder_path, os.path.basename(folder_path)


# ---------------------------------------------------------------------------
# Universal output struct
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Universal container returned by every loader."""

    source_format: str                          # 'TDT' | 'Oxysoft'
    folder_path: str
    folder_name: str

    # timing
    sample_rate: float = 0.0                    # Hz
    num_samples: int   = 0
    duration_s: float  = 0.0                    # seconds

    # signals – shape (num_channels, num_samples)
    signals: Optional[np.ndarray] = None

    # channel metadata
    channel_names: list[str] = field(default_factory=list)
    num_channels: int         = 0

    # events / epocs – list of dicts with at least {'label': str, 'sample': int}
    events: list[dict] = field(default_factory=list)

    # raw header / metadata blob (format-specific)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

class DataFormat(Enum):
    TDT     = auto()
    OXYSOFT = auto()
    UNKNOWN = auto()


# Extensions that are unique to a TDT tank
_TDT_EXTENSIONS = {'.tbk', '.tdx', '.tev', '.tsq', '.sev'}

# Oxysoft TXT exports always contain this string somewhere in the first ~30 lines
_OXYSOFT_HEADER_MARKER = 'Datafile sample rate'


def detect_format(folder_path: str) -> DataFormat:
    """
    Inspect the contents of *folder_path* and return the matching DataFormat.

    Priority: TDT first (proprietary extensions are unambiguous), then
    Oxysoft (look inside .txt files for the header marker).
    """
    entries    = os.listdir(folder_path)
    extensions = {os.path.splitext(e)[1].lower() for e in entries}

    # --- TDT ---
    if extensions & _TDT_EXTENSIONS:
        return DataFormat.TDT

    # --- Oxysoft TXT ---
    txt_files = [e for e in entries if e.lower().endswith('.txt')]
    for fname in txt_files:
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as fh:
                for _ in range(30):
                    if _OXYSOFT_HEADER_MARKER in fh.readline():
                        return DataFormat.OXYSOFT
        except OSError:
            continue

    return DataFormat.UNKNOWN


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_dataset(folder_path: str, fmt: DataFormat | None = None) -> Dataset:
    """
    Load a recording folder and return a Dataset.
    If *fmt* is None, detect_format() is called automatically.
    """
    folder_name = os.path.basename(folder_path.rstrip('/\\'))

    if fmt is None:
        fmt = detect_format(folder_path)

    if fmt is DataFormat.TDT:
        return _load_tdt(folder_path, folder_name)
    elif fmt is DataFormat.OXYSOFT:
        return _load_oxysoft(folder_path, folder_name)
    else:
        raise ValueError(
            f"Could not identify a supported data format in: {folder_path}\n"
            "Expected TDT proprietary files (.Tbk/.tev/…) or an Oxysoft .txt export."
        )


# ---------------------------------------------------------------------------
# TDT loader  – uses your existing functions from the neuroscience lab
# ---------------------------------------------------------------------------

def _load_tdt(folder_path: str, folder_name: str) -> Dataset:
    """
    Load a TDT tank using the existing validate + process pipeline.
    Wraps process_tdt_folder() and maps its output into Dataset.
    """
    # --- validation (your existing function) ---
    valid, msg = processing_TDT.validate_tdt_folder(folder_path)
    if not valid:
        raise ValueError(f"TDT validation failed: {msg}")

    # --- full processing pipeline (your existing function) ---
    result = processing_TDT.process_tdt_folder(folder_path)

    # result keys: x, raw, corr, dff, f0, fs, store, markers
    signals    = result["corr"]                         # final ΔF/F, 1-D array
    signals_2d = signals[np.newaxis, :]                 # → (1, num_samples)
    fs         = float(result["fs"])
    num_samp   = signals.shape[0]

    # markers → unified event list
    # get_event_markers returns [{'time': float, 'label': str, 'color': str}, ...]
    events = []
    for marker in result["markers"]:
        events.append({
            "label":  marker["label"],
            "sample": int(round(marker["time"] * fs)),
        })

    return Dataset(
        source_format = "TDT",
        folder_path   = folder_path,
        folder_name   = folder_name,
        sample_rate   = fs,
        num_samples   = num_samp,
        duration_s    = num_samp / fs if fs > 0 else 0.0,
        signals       = signals_2d,
        channel_names = [result["store"]],
        num_channels  = 1,
        events        = events,
        metadata      = {
            "f0":    result["f0"],
            "raw":   result["raw"],
            "x":     result["x"],
        },
    )


# ---------------------------------------------------------------------------
# Oxysoft / Artinis TXT loader
# ---------------------------------------------------------------------------

_HEADER_KV      = re.compile(r'^([^:\t]+)[:\t]\s*(.*)$')
_LEGEND_MARKERS = ('sample', 'time', '#')


def _load_oxysoft(folder_path: str, folder_name: str) -> Dataset:
    """
    Parse an Oxysoft TXT export folder.
    Multiple .txt files (split recordings) are concatenated in alphabetical order.
    """
    txt_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith('.txt')
    )
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {folder_path}")

    all_signals:   list[np.ndarray] = []
    all_events:    list[dict]       = []
    metadata:      dict             = {}
    channel_names: list[str]        = []
    sample_rate:   float            = 0.0
    first_file = True

    for fname in txt_files:
        fpath = os.path.join(folder_path, fname)
        signals, events, meta, ch_names, fs = _parse_oxysoft_txt(fpath)

        if first_file:
            metadata      = meta
            channel_names = ch_names
            sample_rate   = fs
            first_file    = False

        all_signals.append(signals)
        all_events.extend(events)

    signals_concat            = np.concatenate(all_signals, axis=1)
    num_channels, num_samples = signals_concat.shape
    duration_s = num_samples / sample_rate if sample_rate > 0 else 0.0

    return Dataset(
        source_format = "Oxysoft",
        folder_path   = folder_path,
        folder_name   = folder_name,
        sample_rate   = sample_rate,
        num_samples   = num_samples,
        duration_s    = duration_s,
        signals       = signals_concat,
        channel_names = channel_names,
        num_channels  = num_channels,
        events        = all_events,
        metadata      = metadata,
    )


def _parse_oxysoft_txt(
    filepath: str,
) -> tuple[np.ndarray, list[dict], dict, list[str], float]:
    """
    Parse a single Oxysoft .txt file.

    Returns
    -------
    signals       : np.ndarray  shape (n_channels, n_samples)
    events        : list of {'label': str, 'sample': int}
    metadata      : dict  (header key-value pairs)
    channel_names : list[str]
    sample_rate   : float (Hz)
    """
    metadata:      dict  = {}
    channel_names: list  = []
    data_rows:     list  = []
    events:        list  = []
    sample_rate          = 0.0

    in_header    = True
    legend_found = False
    event_col    = None

    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        for raw_line in fh:
            line  = raw_line.rstrip('\n')
            parts = line.split('\t')

            # ---- header section ----------------------------------------
            if in_header:
                m = _HEADER_KV.match(line)
                if m:
                    key, val = m.group(1).strip(), m.group(2).strip()
                    metadata[key] = val
                    if 'sample rate' in key.lower():
                        try:
                            sample_rate = float(val.split()[0])
                        except (ValueError, IndexError):
                            pass

                # Legend row: first column is 'sample', 'time', or '#'
                if parts and parts[0].strip().lower() in _LEGEND_MARKERS:
                    in_header    = False
                    legend_found = True
                    cols          = [p.strip() for p in parts]
                    event_col     = len(cols) - 1
                    channel_names = cols[1:event_col]
                continue

            # ---- data section ------------------------------------------
            if not legend_found or len(parts) < 2:
                continue

            try:
                row_vals = [
                    float(p) if p.strip() else 0.0
                    for p in parts[1:event_col]
                ]
                data_rows.append(row_vals)
            except ValueError:
                continue

            # event column: non-empty = event label at this sample
            if event_col is not None and event_col < len(parts):
                ev_label = parts[event_col].strip()
                if ev_label:
                    events.append({'label': ev_label, 'sample': len(data_rows) - 1})

    if not data_rows:
        raise ValueError(f"No data rows parsed from {filepath}")

    signals = np.array(data_rows, dtype=np.float64).T  # (channels, samples)
    return signals, events, metadata, channel_names, sample_rate

















