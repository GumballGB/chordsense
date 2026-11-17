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

def detect_start(samples, sample_rate, threshold=0.2, skip_seconds=0.1):
    """
    Detect start of non-silent audio in a recording.

    This is most relevant for actual audio captures. For example,
    a guitar strum does not start at the first sample.

    A skip is also needed to ignore initial silence. but also
    for inconveniences that produces sounds like the start of a guitar strum
    """

    # convert to array (numpy)
    samples = np.asarray(samples)

    # First validate the samples
    if samples.ndim != 1:
        raise ValueError("samples must be 1D")
    if samples.size == 0:
        return 0

    # get the relative threshold percentage of the sample
    threshold *= np.max(np.abs(samples))


    # use numpy to find the start fast
    thresholded_sample = np.flatnonzero(np.abs(samples) > threshold)

    # pick first above threshold
    if thresholded_sample.size > 0:
        start_index = thresholded_sample[0]
    else: # This means no sample passed threshold, silence
        start_index = 0

    # Apply the skip_seconds offset
    start_index = max(start_index + int(skip_seconds * sample_rate), 0)
        
    # If none found, return 0, this could be white noise
    return start_index


def apply_hann(frame):
    """Return (windowed_samples, weights) for later amplitude correction."""

    samples = np.asarray(frame, dtype=float)

    if samples.ndim != 1:
        raise ValueError("frame must be a 1D mono signal")

    # Compute Hann weights for the frame
    weights = np.hanning(samples.size)

    # Apply the window to the samples
    return samples * weights, weights
