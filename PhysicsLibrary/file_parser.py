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

import numpy as np
from PhysicsLibrary import processing_TDT


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
    folder_path:   str
    folder_name:   str

    # timing
    sample_rate: float = 0.0                    # Hz
    num_samples: int   = 0
    duration_s:  float = 0.0                    # seconds

    # signals – shape (num_channels, num_samples)
    signals: Optional[np.ndarray] = None

    # channel metadata
    channel_names: list[str] = field(default_factory=list)
    num_channels:  int       = 0

    # events – list of dicts with at least {'label': str, 'sample': int}
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


_TDT_EXTENSIONS     = {'.tbk', '.tdx', '.tev', '.tsq', '.sev'}
_OXYSOFT_HEADER_MARKER = 'Datafile sample rate'


def detect_format(folder_path: str) -> DataFormat:
    """
    Inspect the contents of *folder_path* and return the matching DataFormat.
    Priority: TDT first (proprietary extensions), then Oxysoft (.txt marker).
    """
    entries    = os.listdir(folder_path)
    extensions = {os.path.splitext(e)[1].lower() for e in entries}

    if extensions & _TDT_EXTENSIONS:
        return DataFormat.TDT

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



def detect_format_file(file_path: str) -> DataFormat:
    """
    Detect the format of a single file (as opposed to a folder).
    Currently supports Oxysoft .txt exports.
    """
    if not file_path.lower().endswith('.txt'):
        return DataFormat.UNKNOWN
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as fh:
            for _ in range(30):
                if _OXYSOFT_HEADER_MARKER in fh.readline():
                    return DataFormat.OXYSOFT
    except OSError:
        pass
    return DataFormat.UNKNOWN


def load_dataset_file(file_path: str) -> Dataset:
    """
    Load a single file and return a Dataset.
    """
    fmt = detect_format_file(file_path)
    if fmt == DataFormat.OXYSOFT:
        folder_name = os.path.splitext(os.path.basename(file_path))[0]
        o2hb, hhb, events, metadata, ch_labels, sample_rate = _parse_oxysoft_txt(file_path)
        n_ch, n_samp = o2hb.shape
        signals      = np.concatenate([o2hb, hhb], axis=0)
        return Dataset(
            source_format = "Oxysoft",
            folder_path   = os.path.dirname(file_path),
            folder_name   = folder_name,
            sample_rate   = sample_rate,
            num_samples   = n_samp,
            duration_s    = n_samp / sample_rate if sample_rate > 0 else 0.0,
            signals       = signals,
            channel_names = [f"{l} O2Hb" for l in ch_labels] + [f"{l} HHb" for l in ch_labels],
            num_channels  = 2 * n_ch,
            events        = events,
            metadata      = {**metadata, "n_channels": n_ch},
        )
    else:
        raise ValueError(f"Unrecognised file format: {file_path}")


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
# TDT loader
# ---------------------------------------------------------------------------

def _load_tdt(folder_path: str, folder_name: str) -> Dataset:
    """
    Load a TDT tank using the existing validate + process pipeline.
    """
    valid, msg = processing_TDT.validate_tdt_folder(folder_path)
    if not valid:
        raise ValueError(f"TDT validation failed: {msg}")

    result     = processing_TDT.process_tdt_folder(folder_path)
    signals    = result["corr"]
    signals_2d = signals[np.newaxis, :]
    fs         = float(result["fs"])
    num_samp   = signals.shape[0]

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
            "f0":  result["f0"],
            "raw": result["raw"],
            "x":   result["x"],
        },
    )


# ---------------------------------------------------------------------------
# Oxysoft / Artinis TXT loader
# ---------------------------------------------------------------------------

def _load_oxysoft(folder_path: str, folder_name: str) -> Dataset:
    """
    Parse an Oxysoft TXT export folder.
    Multiple .txt files are concatenated in alphabetical order.
    """
    txt_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith('.txt')
    )
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {folder_path}")

    all_o2hb:  list[np.ndarray] = []
    all_hhb:   list[np.ndarray] = []
    all_events: list[dict]      = []
    metadata:   dict            = {}
    ch_labels:  list[str]       = []
    sample_rate: float          = 0.0
    first_file = True

    for fname in txt_files:
        fpath = os.path.join(folder_path, fname)
        o2hb, hhb, events, meta, labels, fs = _parse_oxysoft_txt(fpath)

        if first_file:
            metadata    = meta
            ch_labels   = labels
            sample_rate = fs
            first_file  = False

        all_o2hb.append(o2hb)
        all_hhb.append(hhb)
        all_events.extend(events)

    # o2hb / hhb: shape (n_channels, n_samples)
    o2hb_concat = np.concatenate(all_o2hb, axis=1)
    hhb_concat  = np.concatenate(all_hhb,  axis=1)

    n_ch, n_samp = o2hb_concat.shape
    duration_s   = n_samp / sample_rate if sample_rate > 0 else 0.0

    # Stack as (2 * n_channels, n_samples): first all O2Hb, then all HHb
    signals = np.concatenate([o2hb_concat, hhb_concat], axis=0)

    # Channel names: e.g. ['Tx1 O2Hb', 'Tx2 O2Hb', 'Tx3 O2Hb', 'Tx1 HHb', ...]
    o2hb_names = [f"{l} O2Hb" for l in ch_labels]
    hhb_names  = [f"{l} HHb"  for l in ch_labels]

    return Dataset(
        source_format = "Oxysoft",
        folder_path   = folder_path,
        folder_name   = folder_name,
        sample_rate   = sample_rate,
        num_samples   = n_samp,
        duration_s    = duration_s,
        signals       = signals,
        channel_names = o2hb_names + hhb_names,
        num_channels  = 2 * n_ch,
        events        = all_events,
        metadata      = {
            **metadata,
            "n_channels": n_ch,   # how many physical channels (not * 2)
        },
    )


def _parse_oxysoft_txt(
    filepath: str,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict, list[str], float]:
    """
    Parse a single Oxysoft .txt file.

    The Oxysoft format has a Legend block that maps column numbers to
    channel names, followed by a numeric header row (1  2  3 ...) and
    then the data. Row 0 is absolute baseline and is skipped.

    Returns
    -------
    o2hb          : np.ndarray  shape (n_channels, n_samples)
    hhb           : np.ndarray  shape (n_channels, n_samples)
    events        : list of {'label': str, 'sample': int}
    metadata      : dict
    channel_labels: list[str]  e.g. ['Tx1', 'Tx2', 'Tx3']
    sample_rate   : float (Hz)
    """
    metadata:  dict  = {}
    events:    list  = []
    data_rows: list  = []
    sample_rate      = 0.0

    # Will be filled from the Legend block
    # col_index → ('O2Hb' | 'HHb', channel_label)
    col_map: dict[int, tuple[str, str]] = {}
    event_col: Optional[int] = None

    in_legend   = False
    in_data     = False

    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        for raw_line in fh:
            line  = raw_line.rstrip('\n')
            parts = line.split('\t')

            # ---- metadata -----------------------------------------------
            if not in_legend and not in_data:
                if 'sample rate' in line.lower():
                    try:
                        sample_rate = float(parts[1].strip().split()[0])
                    except (IndexError, ValueError):
                        pass

                if line.strip() == 'Legend:':
                    in_legend = True
                    continue

            # ---- legend block -------------------------------------------
            if in_legend:
                # The column-number header row signals end of legend
                stripped = [p.strip() for p in parts]
                if all(p.isdigit() for p in stripped if p):
                    in_legend = False
                    in_data   = True
                    # Build col_map from what we parsed
                    continue

                # Legend rows look like: "2\tRx1 - Tx1 O2Hb (filename)"
                if len(parts) >= 2:
                    try:
                        col_idx  = int(parts[0].strip())
                        col_desc = parts[1].strip()
                    except ValueError:
                        continue

                    if '(Event)' in col_desc or 'Event' in col_desc:
                        event_col = col_idx - 1   # convert to 0-based
                    elif 'O2Hb' in col_desc:
                        # Extract transmitter label e.g. "Tx1" from "Rx1 - Tx1 O2Hb ..."
                        m = re.search(r'Tx\d+', col_desc)
                        label = m.group(0) if m else f"Ch{col_idx}"
                        col_map[col_idx - 1] = ('O2Hb', label)
                    elif 'HHb' in col_desc:
                        m = re.search(r'Tx\d+', col_desc)
                        label = m.group(0) if m else f"Ch{col_idx}"
                        col_map[col_idx - 1] = ('HHb', label)
                    # TSI / sample number → ignored
                continue

            # ---- data section -------------------------------------------
            if not in_data or len(parts) < 2:
                continue

            try:
                row = [float(p) if p.strip() else 0.0 for p in parts]
            except ValueError:
                continue

            # Skip row 0 — absolute baseline, not delta values
            sample_num = int(row[0])
            if sample_num == 0:
                continue

            data_rows.append(row)

            # Events
            if event_col is not None and event_col < len(row):
                ev = parts[event_col].strip() if event_col < len(parts) else ''
                if ev and ev != '0':
                    events.append({'label': ev, 'sample': len(data_rows) - 1})

    if not data_rows:
        raise ValueError(f"No data rows parsed from {filepath}")

    data = np.array(data_rows, dtype=np.float64)  # (n_samples, n_cols)

    # Sort col_map into ordered O2Hb and HHb columns
    o2hb_cols = sorted(
        [(idx, label) for idx, (kind, label) in col_map.items() if kind == 'O2Hb'],
        key=lambda x: x[1]
    )
    hhb_cols = sorted(
        [(idx, label) for idx, (kind, label) in col_map.items() if kind == 'HHb'],
        key=lambda x: x[1]
    )

    channel_labels = [label for _, label in o2hb_cols]

    o2hb = np.array([data[:, idx] for idx, _ in o2hb_cols])  # (n_ch, n_samp)
    hhb  = np.array([data[:, idx] for idx, _ in hhb_cols])   # (n_ch, n_samp)

    return o2hb, hhb, events, metadata, channel_labels, sample_rate