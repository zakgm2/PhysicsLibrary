import numpy as np


def visibility_model(beta, A, V, beta_c, Period):
    """
    Parameters
    ----------
    beta   : Angle of the polarizer
    A      : Amplitude of the sinusoid
    V      : Visibility
    beta_c : Center or angular shift
    Period : Period of the wave

    Returns
    -------
    Model for visibility of photon entanglement
    """
    return (A / 2) * (1 - V * np.sin((beta - beta_c) / Period))


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