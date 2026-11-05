# Fast Fourier Transform (FFT) implementation in Python

import numpy as np


def fft(signal):
    """
    Compute the Fast Fourier Transform (FFT) of a 1D signal.

    This implementation uses the radix-2 Cooley-Tukey algorithm and expects
    the input length to be a power of two.

    Parameters:
    signal (array-like): Real or complex-valued 1D input samples.

    Returns:
    numpy.ndarray: Full complex FFT coefficients with length N.

    Raises:
    ValueError: If the input is not 1D or the length is not a power of two.
    """

    # Convert the input signal to a numpy array of complex type
    x = np.asarray(signal, dtype=complex)

    if x.ndim != 1:
        raise ValueError("fft expects a 1D input signal")

    N = x.size
    if N == 0:
        return np.array([], dtype=complex)

    if N & (N - 1):
        raise ValueError("fft input length must be a power of two")

    if N == 1:
        return x.copy()

    # Recursive FFT implementation
    even = fft(x[::2])
    odd = fft(x[1::2])

    # Compute the twiddle factors and combine the results
    twiddle = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd

    return np.concatenate((even + twiddle, even - twiddle))