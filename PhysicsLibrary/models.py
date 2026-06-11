import numpy as np

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

