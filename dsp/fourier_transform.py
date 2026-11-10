# File just for processing the fourier transform
# In all honesty, the stack will just use numpy's FFT implementation,
# but this is a good exercise to understand how the FFT works.

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


def amplitude_spectrum(X, n_samples=None, window=None):
    """
    Compute a single-sided amplitude spectrum from full FFT or rfft bins.

    Parameters:
    X (array-like): Full complex DFT/FFT coefficients, or rfft coefficients.

    n_samples (int | None): Original signal length. Required for rfft output,
        since its length alone cannot distinguish odd and even input lengths.
        Defaults to len(X) for compatibility with full FFT callers.

    window (array-like | None): The actual window weights applied before the
        transform, with length n_samples. Scaling divides by their sum to
        correct for coherent gain. Without a window, scaling divides by N.

    DC and Nyquist are exceptions to doubling in the single-sided spectrum.
    - DC: keep its normalized amplitude unchanged; do not double it.
    - Nyquist: exists only when N is even and is not doubled either.

    For 8 real audio samples, the full FFT has these 8 bins;
    [ DC, +f1, +f2, +f3, Nyquist, -f3, -f2, -f1 ]

    Returns:
    numpy.ndarray: Single-sided amplitude spectrum scaled so that a sinusoid
        with amplitude A has peak amplitude A when it lands exactly on a bin.
    """

    # X is represented in the frequency domain
    X = np.asarray(X)

    # Check that X is 1D and has expected size
    if X.ndim != 1:
        raise ValueError("FFT coefficients must be 1D")

    # Determine the original number of audio samples
    N = X.size if n_samples is None else n_samples
    if not isinstance(N, (int, np.integer)) or N < 0:
        raise ValueError("n_samples must be a nonnegative integer")
    expected_sizes = (N, N // 2 + 1) if N else (0,)

    # Check if X matches expected sizes
    if X.size not in expected_sizes:
        raise ValueError("FFT coefficient count does not match n_samples")

    # The scale is the sum of window weights or N if no window
    scale = N

    if window is not None:
        weights = np.asarray(window, dtype=float)
        if weights.ndim != 1 or weights.size != N:
            raise ValueError("window must be 1D with length n_samples")
        if not np.all(np.isfinite(weights)):
            raise ValueError("window must contain only finite weights")
        scale = weights.sum()

    # edge case: if the input is empty, return an empty array
    if N == 0:
        return np.array([])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("window sum must be positive and finite")

    # Keep non-negative frequencies.
    # Nyquist frequency is included if N is even, otherwise it is not.
    # We double the amplitudes except DC and Nyquist, because of single-sided spectrum
    amps = np.abs(X[: N // 2 + 1]) / scale

    # Double only the frequencies that have a negative-frequency partner.
    if N % 2 == 0:
        amps[1:-1] *= 2
    else:
        amps[1:] *= 2

    return amps
