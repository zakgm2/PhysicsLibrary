"""
models.py
---------
Parametric model functions for curve fitting via scipy.optimize.curve_fit.

Each function follows the signature  f(x, *params) -> y  so they can be
passed directly to fit_model_to_segment in analysis.py.
"""

import numpy as np


def visibility_model(beta, a, v, beta_c, period):
    """
    Photon-entanglement visibility model.

    Parameters
    ----------
    beta   : array  Polarizer angle (degrees or radians)
    a      : float  Amplitude of the sinusoid
    v      : float  Visibility (0–1)
    beta_c : float  Angular offset / centre
    period : float  Period of the oscillation

    Returns
    -------
    array
    """
    return (a / 2) * (1 - v * np.sin((beta - beta_c) / period))


def double_exponential_model(x, a, b, c, d, k):
    """
    Physical model of fluorophore bleaching.
    y = a*exp(-b*x) + c*exp(-d*x) + k
    """
    return a * np.exp(-b * x) + c * np.exp(-d * x) + k


def linear_model(x, m, b):
    """y = mx + b"""
    return m * x + b


def single_exponential_model(x, a, b, c):
    """y = a * exp(-b*x) + c"""
    return a * np.exp(-b * x) + c


def exponential_rise_model(x, a, b, c):
    """y = a * (1 - exp(-b*x)) + c   — rising exponential (e.g. venous occlusion)"""
    return a * (1 - np.exp(-b * x)) + c


def gaussian_model(x, a, mu, sigma):
    """y = a * exp(-(x-mu)^2 / (2*sigma^2))"""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def sinusoidal_model(x, a, f, phi, c):
    """y = a * sin(2*pi*f*x + phi) + c"""
    return a * np.sin(2 * np.pi * f * x + phi) + c