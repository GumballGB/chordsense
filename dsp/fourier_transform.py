# File just for processing the fourier transform

import numpy as np

def dft(signal):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D signal.
    
    Parameters:
    signal: array-like, it is the input signal, it is not aware of the sampling rate, 
    it is just a 1D array of samples.

    Returns:
    numpy.ndarray: Full complex DFT coefficients X[k] for k = 0..N-1.
    """

    # Convert the input signal to a numpy array
    signal = np.asarray(signal)

    # Get the number of samples in the signal
    N = signal.size

    # Initialize an array to hold the DFT results
    X = np.zeros(N, dtype=complex)

    # Compute the DFT using the formula
    for k in range(N):

        # rectangular summation for the DFT
        total = 0 + 0j

        # Compute the k-th frequency bin
        for n in range(N):
            # Compute the angle for the complex exponential
            angle = -2j * np.pi * k * n / N
            total += signal[n] * np.exp(angle)

        X[k] = total

    return X


def amplitude_spectrum(X):
    """
    Compute the single-sided amplitude spectrum from full complex DFT bins.

    Parameters:
    X (array-like): Full complex DFT coefficients with length N.

    Returns:
    numpy.ndarray: Single-sided amplitude spectrum scaled so that a sinusoid
        with amplitude A has peak amplitude A when it lands exactly on a bin.
    """

    X = np.asarray(X)
    N = X.size

    # edge case: if the input is empty, return an empty array
    if N == 0:
        return np.array([])

    # Keep non-negative frequencies.
    # Nyquist frequency is included if N is even, otherwise it is not.
    if N % 2 == 0:
        amps = np.abs(X[: N // 2 + 1]) * (2.0 / N)
        amps[0] *= 0.5
        amps[-1] *= 0.5
    else:
        amps = np.abs(X[: N // 2 + 1]) * (2.0 / N)
        amps[0] *= 0.5
    
    return amps