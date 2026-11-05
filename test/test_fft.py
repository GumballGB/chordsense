# Test our homegrown FFT implementation against the DFT implementation. The FFT should produce the same results as the DFT, but much faster.
import numpy as np
from dsp.fourier_transform import dft
from dsp.fft import fft

def test_fft_vs_dft():
    """
    Test that our FFT implementation produces the same results as the DFT implementation.
    This test generates a sine wave signal, computes its DFT and FFT, and compares the results.
    """
    
    # Test parameters
    frequency = 440  # A4 note
    sample_rate = 4096
    n_samples = 1024  # Power of two for radix-2 FFT
    duration = n_samples / sample_rate

    # Generate a sine wave signal
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)

    # Compute DFT and FFT
    dft_result = dft(signal)
    fft_result = fft(signal)

    # Compare the results
    assert np.allclose(dft_result, fft_result, atol=1e-9), "FFT and DFT results do not match!"