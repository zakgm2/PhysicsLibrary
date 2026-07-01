"""
loaders/tdt_loader.py
-----------------------
Wraps processing_TDT's validate + process pipeline and packages the
result into a universal Dataset.
"""

import numpy as np

from .. import processing_TDT
from ..dataset import Dataset


def load_tdt(folder_path: str, folder_name: str) -> Dataset:
    """Load a TDT tank using the existing validate + process pipeline."""
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
