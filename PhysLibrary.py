import tkinter as tk
from tkinter import filedialog
import numpy as np
import tdt
from scipy.optimize import curve_fit

###
### This is my little library for analysis that I update whenever I need to 
###


def choose_file():
    """
    Opens a directory selection dialog

    Behavior
    --------
    - Opens a Tkinter folder selection dialog
    - If valid:
        * Updates global `folder_path`
        * Updates GUI window title with dataset name
        * Displays success notification
    - If invalid:
        * Displays error message to user
        * Prevents further processing

    Returns
    -------
    File Path

    Notes
    -----
    - This is the primary entry point for loading datasets
    - Acts as a safety gate before any processing occurs
    - Ensures only valid structures are passed
    """
    global folder_path

    root = tk.Tk()
    root.withdraw()          # hides main window
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("All files", "*.*")]
    )

    root.destroy()

    if not file_path:
        print("No file selected")
        return

    print("Selected:", file_path)
    return file_path

def visibility_model(beta, A, V, beta_c, Period):    
    """

    Parameters
    ----------
    beta : Angle of the polarizer 
    A : Amplitude of the sinusoid.
    V : Visibility
    beta_c : Center or angular shift.
    Period : Period of the wave.

    Returns
    -------
    Model for visibility of photon entanglment ???

    """
    return (A/2)*(1 - V*np.sin((beta - beta_c)/Period))

def double_exponential_model(x, a, b, c, d, k):
    """
    The physical model of fluorophore bleaching

    Behavior
    --------
    Computes a double exponential in x with parameters a, b, c, d and k 

    Returns
    -------
    Double exponential 

    Notes
    -----
    None
    """
    return a * np.exp(-b * x) + c * np.exp(-d * x) + k

def get_tdt_struct(path):
    "Makes a Struct From the Binary TDT folder"
    data = tdt.read_block(path)
    if data is None:
        raise Exception("TDT returned an empty object.")
    
    #This is the raw data
    return data

def get_plot_data(data, store_name, channel=0, max_points=5000):
    "Get Data From Struct"
    stream = data.streams[store_name]
    fs = stream.fs  # Extracting the actual sampling frequency
    
    data_2d = np.atleast_2d(stream.data)
    if channel >= data_2d.shape[0]:
        channel = 0
        
    y_full = data_2d[channel, :]
    ds_factor = max(1, len(y_full) // max_points)
    y = y_full[::ds_factor]
    
    # Generate time axis using the EXACT fs from the file
    x = np.arange(len(y)) * (ds_factor / fs)
    
    return x, y, fs

def correct_bleaching(y, fs):
    "Corrects Bleaching according to literature-stated decay"
    x = np.arange(len(y)) / fs
    
    #1. THE RIGID MASK: Only look at the "Top" 50% of the signal
    # This ensures the 0s are invisible to the math
    threshold = np.median(y) 
    mask = y > threshold
    
    x_fit = x[mask]
    y_fit = y[mask]

    if len(y_fit) < 100:
        return y, np.zeros_like(y)

    #2. SMART INITIAL GUESS: Instead of fixed numbers, we look at the data
    #k (baseline) should be near the end of the recording
    k_guess = np.percentile(y_fit, 10) 
    
    #Total Amplitude to be explained by the decay
    total_amp = np.max(y_fit) - k_guess
    
    #p0 = [Amp_fast, Decay_fast, Amp_slow, Decay_slow, Baseline]
    p0 = [total_amp*0.6, 0.05, total_amp*0.4, 0.0001, k_guess]
    
    #3. BOUNDS: Prevent the "Drop to Zero"
    # We force the baseline 'k' to be at least the bottom of our HIGH signal
    lower_bounds = [0, 0, 0, 0, k_guess * 0.8]
    upper_bounds = [np.inf, 1, np.inf, 0.1, np.max(y_fit)]

    try:
        popt, _ = curve_fit(double_exponential_model, x_fit, y_fit, p0=p0, 
                            bounds=(lower_bounds, upper_bounds), maxfev=10000)
        trend = double_exponential_model(x, *popt)
    except:
        # If double-exp fails, fit a single exponential (simpler/more stable)
        coeffs = np.polyfit(x_fit, np.log(np.maximum(y_fit, 1e-6)), 1)
        trend = np.exp(np.polyval(coeffs, x))

    #4. RESTORE: Keep the signal at the "High" mean
    corrected = y - trend + np.mean(y_fit)
    
    return corrected, trend

def get_event_markers(data):
    """
    Returns a list of dictionaries for every note found in the TDT file.
    """
    if not hasattr(data.epocs, 'Note'):
        return []
    
    notes = data.epocs.Note.notes
    onsets = data.epocs.Note.onset
    
    markers = []
    # Simple color map for different notes
    color_map = {'Clap': 'red', 'Sucrose': 'green', 'Stop': 'blue'}
    
    for n, t in zip(notes, onsets):
        note_str = n.decode() if isinstance(n, bytes) else str(n)
        note_str = note_str.strip()
        
        markers.append({
            'time': t,
            'label': note_str,
            'color': color_map.get(note_str, 'black') # Default to black if unknown
        })
    return markers