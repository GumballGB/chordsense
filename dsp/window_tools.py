"""
Framing and Hann windowing for mono audio.

A frame is a segment of audio. A Hann window fades that segment's edges toward
zero before an FFT to reduce spectral leakage. 

The pipeline calls these steps
explicitly after converting audio to mono with dsp.mono_tools.to_mono.

The window's coherent gain must be considered when analyzing amplitudes.
"""

import numpy as np


def take_frame_mono(signal, frame_size=44100):
    """Take up to frame_size samples from already-mono audio, without resampling."""

    # Convert to array and check mono
    samples = np.asarray(signal, dtype=float)

    # Check mono
    if samples.ndim != 1:
        raise ValueError("signal must be a 1D mono signal")
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")

    # Take the first frame_size samples
    return samples[:frame_size]


def apply_hann(frame):
    """Return (windowed_samples, weights) for later amplitude correction."""

    samples = np.asarray(frame, dtype=float)

    if samples.ndim != 1:
        raise ValueError("frame must be a 1D mono signal")

    # Compute Hann weights for the frame
    weights = np.hanning(samples.size)

    # Apply the window to the samples
    return samples * weights, weights
