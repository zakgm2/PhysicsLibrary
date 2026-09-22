"""
analysis/curve_fit.py
-----------------------
Two-point linear slope between click-selected indices, and general
model fitting (any model_fn/p0_fn pair — see PhysicsLibrary.models for
the model functions themselves) via scipy curve_fit.
"""

import numpy as np
from scipy.optimize import curve_fit


def compute_slope_segment(x_data, y_data, p1_idx, p2_idx, padding_pct=0.05):
    """
    Least-squares linear regression slope between two index boundaries.

    Parameters
    ----------
    x_data      : array
    y_data      : array
    p1_idx      : int
    p2_idx      : int
    padding_pct : float   visual context padding

    Returns
    -------
    dict with slope, intercept, crop_x, crop_y, x1, y1, x2, y2
    """
    idx1, idx2 = sorted([p1_idx, p2_idx])

    fit_x = x_data[idx1:idx2 + 1]
    fit_y = y_data[idx1:idx2 + 1]

    if len(fit_x) < 2:
        slope, intercept = 0.0, 0.0
    else:
        slope, intercept = np.polyfit(fit_x, fit_y, 1)

    x1, y1 = fit_x[0],  fit_y[0]
    x2, y2 = fit_x[-1], fit_y[-1]

    pad        = max(5, int(len(x_data) * padding_pct))
    start_idx  = max(0, idx1 - pad)
    end_idx    = min(len(x_data) - 1, idx2 + pad)

    return {
        'slope':     slope,
        'intercept': intercept,
        'crop_x':    x_data[start_idx:end_idx + 1],
        'crop_y':    y_data[start_idx:end_idx + 1],
        'x1': x1, 'y1': y1,
        'x2': x2, 'y2': y2,
    }


def fit_model_to_segment(x_seg, y_seg, model_fn, p0_fn):
    """
    Fit a model function to a data segment using scipy curve_fit.

    Parameters
    ----------
    x_seg    : array
    y_seg    : array
    model_fn : callable   f(x, *params) -> y
    p0_fn    : callable   f(x_seg, y_seg) -> list of initial guesses

    Returns
    -------
    dict with popt, y_fit, r2, success, error
    """
    try:
        p0      = p0_fn(x_seg, y_seg)
        popt, _ = curve_fit(model_fn, x_seg, y_seg, p0=p0, maxfev=10000)
        y_fit   = model_fn(x_seg, *popt)
        ss_res  = np.sum((y_seg - y_fit) ** 2)
        ss_tot  = np.sum((y_seg - y_seg.mean()) ** 2)
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"popt": popt, "y_fit": y_fit, "r2": r2,
                "success": True, "error": None}
    except Exception as e:
        return {"popt": None, "y_fit": np.zeros_like(y_seg),
                "r2": 0.0, "success": False, "error": str(e)}
