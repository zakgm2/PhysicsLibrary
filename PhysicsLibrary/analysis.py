"""
analysis.py
-----------
Format-agnostic analysis routines for Physics Analysis GUI.

Currently:
  - Z-Score PETH (get_zscore_slice, smooth_signal, bin_for_heatmap)
"""

import numpy as np
import pandas as pd

def get_zscore_slice(time_array, signal, center_t, window=30):
    """
    Extract and z-score a time window around an event.

    Parameters
    ----------
    time_array : array
    signal : array
    center_t : float
        Event time in seconds
    window : float
        Total window size in seconds

    Returns
    -------
    (time segment, z-scored signal)
    """
    half_win  = window / 2
    start_idx = np.searchsorted(time_array, center_t - half_win)
    end_idx   = np.searchsorted(time_array, center_t + half_win)

    seg_y = signal[start_idx:end_idx]
    seg_x = time_array[start_idx:end_idx]

    seg_y = np.clip(seg_y, -5, 5)

    baseline_end    = len(seg_y) // 2
    baseline_period = seg_y[:baseline_end]
    mu  = np.mean(baseline_period)
    std = np.std(baseline_period)

    if std < 1e-6:
        return seg_x, np.zeros_like(seg_y)

    return seg_x, (seg_y - mu) / std


def smooth_signal(data, fs, window_sec=0.5):
    """
    Moving average smoothing filter.

    Parameters
    ----------
    data : array
    fs : float
        Sampling frequency in Hz
    window_sec : float
        Smoothing window in seconds

    Returns
    -------
    array
        Smoothed signal
    """
    window_size = int(fs * window_sec)
    if window_size % 2 == 0:
        window_size += 1
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def bin_for_heatmap(z_seg, num_bins=300):
    """
    Bin a signal into equal segments for heatmap plotting.

    Parameters
    ----------
    z_seg : array
    num_bins : int

    Returns
    -------
    array
        Binned signal
    """
    if z_seg is None or len(z_seg) == 0:
        return np.zeros(num_bins)
    bin_edges = np.linspace(0, len(z_seg), num_bins + 1).astype(int)
    return np.array([np.mean(z_seg[bin_edges[i]:bin_edges[i+1]]) for i in range(num_bins)])


def compute_fft_slice(time_array, signal, center_t, fs, window=30):
    """
    Extract a time window around center_t and compute its FFT.

    Applies mean removal and linear detrending before FFT to eliminate
    the DC spike and slow drift, making physiological frequencies
    (breathing ~0.3 Hz, heart rate ~1 Hz) visible.

    Parameters
    ----------
    time_array : array
    signal : array
    center_t : float
        Center time in seconds
    fs : float
        Sampling frequency in Hz
    window : float
        Total window size in seconds

    Returns
    -------
    freqs : array
        Positive frequencies in Hz
    power : array
        Power spectral density (magnitude squared)
    seg_x : array
        Time array of the extracted segment
    seg_y : array
        Signal values of the extracted segment
    """
    from scipy.signal import detrend

    half_win  = window / 2
    start_idx = np.searchsorted(time_array, center_t - half_win)
    end_idx   = np.searchsorted(time_array, center_t + half_win)

    seg_y = signal[start_idx:end_idx]
    seg_x = time_array[start_idx:end_idx]

    if len(seg_y) < 4:
        return np.array([]), np.array([]), seg_x, seg_y

    # 1. Remove linear trend (kills DC + slow drift)
    seg_y = detrend(seg_y, type='linear')

    # 2. Remove any residual mean
    seg_y = seg_y - np.mean(seg_y)

    # 3. Hanning window to reduce spectral leakage
    windowed = seg_y * np.hanning(len(seg_y))

    n     = len(windowed)
    fft_y = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = (np.abs(fft_y) ** 2) / n

    return freqs, power, seg_x, seg_y


def annotate_fft_peaks(ax_f, freqs, power, color, n_peaks=3):
    """
    Find top N peaks in a power spectrum and annotate them with
    frequency and BPM labels directly on the axes.

    Parameters
    ----------
    ax_f   : matplotlib Axes
    freqs  : array
    power  : array
    color  : str
    n_peaks: int
    """
    from scipy.signal import find_peaks

    mask     = freqs >= 0.05
    f_m      = freqs[mask]
    p_m      = power[mask]
    if len(p_m) < 3:
        return
    min_prom = 0.05 * p_m.max()
    peaks, _ = find_peaks(p_m, prominence=min_prom)
    if len(peaks) == 0:
        return
    top = sorted(peaks, key=lambda i: p_m[i], reverse=True)[:n_peaks]
    for idx in top:
        freq = f_m[idx]
        pwr  = p_m[idx]
        bpm  = freq * 60
        ax_f.annotate(
            f"{freq:.2f} Hz\n({bpm:.0f} bpm)",
            xy=(freq, pwr),
            xytext=(freq + 0.05, pwr * 0.92),
            fontsize=7, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
        )
        ax_f.axvline(freq, color=color, lw=0.7, linestyle=':', alpha=0.5)
        
def compute_slope_segment(x_data, y_data, p1_idx, p2_idx, padding_pct=0.05):
    """
    Calculates the slope between two data indices and returns cropped segments 
    of the original datasets centered around the selection.

    Parameters
    ----------
    x_data : array
    y_data : array
    p1_idx : int
        Index of the first selected point
    p2_idx : int
        Index of the second selected point
    padding_pct : float
        Percentage of the total array length to add as visual context padding

    Returns
    -------
    dict
        A structured container holding:
        - 'slope': float
        - 'crop_x': array
        - 'crop_y': array
        - 'x1', 'y1', 'x2', 'y2': snapped point coordinates
    """
    # Sort indices chronologically to avoid negative dx from backwards clicks
    idx1, idx2 = sorted([p1_idx, p2_idx])
    
    x1, y1 = x_data[idx1], y_data[idx1]
    x2, y2 = x_data[idx2], y_data[idx2]
    
    # Calculate geometric slope (rise over run)
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
    
    # Calculate situational padding bounds
    window_padding = max(5, int(len(x_data) * padding_pct))
    start_idx = max(0, idx1 - window_padding)
    end_idx = min(len(x_data) - 1, idx2 + window_padding)
    
    return {
        'slope': slope,
        'crop_x': x_data[start_idx:end_idx+1],
        'crop_y': y_data[start_idx:end_idx+1],
        'x1': x1, 'y1': y1,
        'x2': x2, 'y2': y2
    }