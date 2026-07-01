"""
processing_TDT.py
-----------------
Signal processing pipeline for TDT (Tucker-Davis Technologies) fibre photometry data.

Handles loading, bleaching correction, denoising, and event marker extraction.
Depends on the `tdt` Python SDK for reading tank files.
"""

import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import tdt

from .models import double_exponential_model as double_exponential


def validate_tdt_folder(path):
    """
    Validate whether a directory contains a TDT recording block.

    Parameters
    ----------
    path : str
        Path to the TDT data folder.

    Returns
    -------
    tuple
        (bool, str)
        - True + folder name if valid
        - False + error message if invalid
    """
    if not path:
        return False, "No folder selected."

    has_tbk = any(fname.endswith('.Tbk') for fname in os.listdir(path))

    if has_tbk:
        return True, os.path.basename(path)
    else:
        return False, "Invalid Folder: No TDT block files (.Tbk) found."


def process_tdt_folder(folder_path):
    """
    Full photometry processing pipeline for a TDT recording.

    Steps:
    1. Load TDT block
    2. Extract 465 nm (signal) and optional 415 nm (reference)
    3. Perform regression-based motion correction
    4. Correct photobleaching trend
    5. Compute ΔF/F
    6. Denoise signal
    7. Extract event markers

    Parameters
    ----------
    folder_path : str
        Path to TDT recording folder.

    Returns
    -------
    dict
        Processed signals and metadata:
        - x: time vector
        - raw: corrected fluorescence signal
        - corr: final ΔF/F (denoised)
        - dff: same as corr (alias)
        - f0: bleaching baseline
        - fs: sampling frequency
        - store: signal label
        - markers: behavioral event markers
    """
    data_struct = get_tdt_struct(folder_path)
    streams = data_struct.streams.keys()

    name_465 = next((s for s in streams if '465' in s), None)
    name_415 = next((s for s in streams if '415' in s), None)

    if not name_465:
        raise ValueError("No 465 signal found")

    x, y_465, fs = get_plot_data(data_struct, name_465)

    if name_415:
        _, y_415, _ = get_plot_data(data_struct, name_415)
        p = np.polyfit(y_415, y_465, 1)
        y_fit = np.polyval(p, y_415)
        y_final = y_465 - y_fit
        display_name = f"Corrected {name_465} (via {name_415})"
    else:
        y_final = y_465
        display_name = f"{name_465} (Uncorrected)"

    _, trend = correct_bleaching(y_final, fs)

    f0  = np.maximum(trend, 1e-6)
    dff = (y_final - f0) / f0
    dff = denoise_signal(dff, fs, cutoff=5)

    return {
        "x":       x,
        "raw":     y_final,
        "corr":    dff,
        "dff":     dff,
        "f0":      f0,
        "fs":      fs,
        "store":   display_name,
        "markers": get_event_markers(data_struct),
    }


def get_tdt_struct(path):
    """
    Load a Tucker-Davis Technologies (TDT) recording block.

    Parameters
    ----------
    path : str
        Folder containing TDT data.

    Returns
    -------
    object
        Parsed TDT data structure.
    """
    data = tdt.read_block(path)
    if data is None:
        raise Exception("TDT returned an empty object.")
    return data


def get_plot_data(data, store_name, channel=0, max_points=None):
    """
    Extract time-series data from a TDT stream.

    Parameters
    ----------
    data : object
        TDT data structure
    store_name : str
        Stream name (e.g., 'x465A')
    channel : int
        Channel index
    max_points : int or None
        Optional downsampling limit

    Returns
    -------
    tuple
        (time array, signal array, sampling frequency)
    """
    stream = data.streams[store_name]
    fs     = stream.fs

    data_2d = np.atleast_2d(stream.data)
    if channel >= data_2d.shape[0]:
        channel = 0

    y_full = data_2d[channel, :]

    if max_points:
        ds_factor = max(1, len(y_full) // max_points)
        y = y_full[::ds_factor]
        x = np.arange(len(y)) * (ds_factor / fs)
    else:
        y = y_full
        x = np.arange(len(y)) / fs

    return x, y, fs


def correct_bleaching(y, fs):
    """
    Estimate and correct photobleaching using masked curve fitting.

    Returns
    -------
    corrected : array
        Bleaching-corrected signal
    trend : array
        Estimated baseline trend
    """
    x = np.arange(len(y)) / fs

    threshold = np.median(y)
    mask = y > threshold
    x_fit, y_fit = x[mask], y[mask]

    if len(y_fit) < 100:
        return y, np.zeros_like(y)

    k_guess   = np.percentile(y_fit, 10)
    total_amp = np.max(y_fit) - k_guess
    p0        = [total_amp * 0.6, 0.05, total_amp * 0.4, 0.0001, k_guess]
    lower     = [0, 0, 0, 0, k_guess * 0.8]
    upper     = [np.inf, 1, np.inf, 0.1, np.max(y_fit)]

    try:
        popt, _ = curve_fit(double_exponential, x_fit, y_fit, p0=p0,
                            bounds=(lower, upper), maxfev=10000)
        trend = double_exponential(x, *popt)
    except Exception:
        # curve_fit failed — fall back to log-linear fit as a rough trend estimate
        import warnings
        warnings.warn(
            "correct_bleaching: double-exponential fit failed; falling back to log-linear trend.",
            RuntimeWarning, stacklevel=2,
        )
        coeffs = np.polyfit(x_fit, np.log(np.maximum(y_fit, 1e-6)), 1)
        trend  = np.exp(np.polyval(coeffs, x))

    corrected = y - trend + trend[0]
    return corrected, trend


def get_event_markers(data):
    """
    Extract behavioral event markers from TDT epoc data.

    Returns
    -------
    list of dict
        Each dict contains:
        - time
        - label
        - color
    """
    if not hasattr(data.epocs, 'Note'):
        return []

    notes  = data.epocs.Note.notes
    onsets = data.epocs.Note.onset

    # Experiment-specific label→colour mapping; unknown labels default to black.
    color_map = {'Clap': 'red', 'Sucrose': 'green', 'Stop': 'blue'}
    markers   = []

    for n, t in zip(notes, onsets):
        note_str = n.decode() if isinstance(n, bytes) else str(n)
        note_str = note_str.strip()
        markers.append({
            'time':  t,
            'label': note_str,
            'color': color_map.get(note_str, 'black'),
        })
    return markers


def denoise_signal(signal, fs, cutoff=5, order=2):
    """
    Low-pass Butterworth filter for ΔF/F signals.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz
    order : int
        Filter order

    Returns
    -------
    array
        Filtered signal
    """
    nyquist       = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a          = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)