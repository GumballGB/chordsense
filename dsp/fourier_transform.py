# File just for processing the fourier transform

import numpy as np

def dft(signal):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D signal.

    Parameters:
    signal (array-like): Input signal to transform.

    Returns:
    numpy.ndarray: Magnitude spectrum of the signal, limited to the first
        half of the frequency bins.
    """

    signal = np.asarray(signal)
    N = signal.size
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        total = 0 + 0j

        for n in range(N):
            angle = -2j * np.pi * k * n / N
            total += signal[n] * np.exp(angle)

        X[k] = total

    return np.abs(X[:N // 2])