"""
dataset.py
----------
The universal Dataset container returned by every loader, plus format
detection. No parsing logic lives here — that's in loaders/. No GUI
concerns either — folder/file selection is a caller responsibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


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


_TDT_EXTENSIONS        = {'.tbk', '.tdx', '.tev', '.tsq', '.sev'}
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
