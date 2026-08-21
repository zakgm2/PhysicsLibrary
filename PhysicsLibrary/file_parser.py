"""
file_parser.py
--------------
Top-level dispatcher: detects the recording format inside a selected
folder/file and routes to the matching loader, returning a universal
Dataset struct.

  - TDT  (Tucker-Davis Technologies) – loaders/tdt_loader.py
  - Oxysoft / Artinis (Oxymon, OctaMon, PortaMon …) – loaders/oxysoft_loader.py

Format detection, the Dataset struct, and folder selection live in
dataset.py. The .pt2 EFNMR/MRI image format is a standalone parser in
loaders/pt2_loader.py (it returns a raw image array, not a Dataset).

Usage
-----
    from file_parser import detect_format, load_dataset, DataFormat

    fmt = detect_format(folder_path)
    dataset = load_dataset(folder_path, fmt)
"""

from __future__ import annotations

import os
from typing import Optional

from .dataset import Dataset, DataFormat, detect_format, detect_format_file
from .loaders.tdt_loader import load_tdt
from .loaders.oxysoft_loader import load_oxysoft, load_oxysoft_file


def load_dataset_file(file_path: str) -> Dataset:
    """Load a single file and return a Dataset."""
    fmt = detect_format_file(file_path)
    if fmt == DataFormat.OXYSOFT:
        return load_oxysoft_file(file_path)
    raise ValueError(f"Unrecognised file format: {file_path}")


def load_dataset(folder_path: str, fmt: Optional[DataFormat] = None,
                  regression_method: str = "ransac") -> Dataset:
    """
    Load a recording folder and return a Dataset.
    If *fmt* is None, detect_format() is called automatically.

    regression_method : {"ransac", "huber", "ols"}
        TDT-only — forwarded to load_tdt()/process_tdt_folder() for the
        isosbestic-vs-signal motion-correction regression. Ignored for
        Oxysoft, which has no such correction step.
    """
    folder_name = os.path.basename(folder_path.rstrip('/\\'))

    if fmt is None:
        fmt = detect_format(folder_path)

    if fmt is DataFormat.TDT:
        return load_tdt(folder_path, folder_name, regression_method=regression_method)
    elif fmt is DataFormat.OXYSOFT:
        return load_oxysoft(folder_path, folder_name)
    else:
        raise ValueError(
            f"Could not identify a supported data format in: {folder_path}\n"
            "Expected TDT proprietary files (.Tbk/.tev/…) or an Oxysoft .txt export."
        )
