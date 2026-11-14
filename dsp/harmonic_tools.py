# Simple timbre shaping for synthesized harmonics.

import numpy as np


def dampen_harmonics(harmonic, times, harmonic_number, decay_rate=2.0):
    """Make one harmonic quieter and let higher harmonics fade faster.

    harmonic is the waveform for this harmonic.
    times is the array of sample times.
    harmonic_number is 1 for the fundamental.
    decay_rate is how fast it fades.

    Apply this to each harmonic before summing them. harmonic_number starts at
    1 for the fundamental. times contains sample times in seconds. decay_rate
    controls the exponential fade in inverse seconds; zero disables the fade.
    The initial amplitude weight is 1 / harmonic_number. This is a simple
    adjustable approximation, not a physical model of a guitar string.
    """

    # Basic checks and conversions
    samples = np.asarray(harmonic, dtype=float)
    times = np.asarray(times, dtype=float)
    if samples.ndim != 1 or times.shape != samples.shape:
        raise ValueError("harmonic and times must be matching 1D arrays")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(times)):
        raise ValueError("harmonic and times must contain only finite values")
    if np.any(times < 0):
        raise ValueError("times must be nonnegative")
    if not isinstance(harmonic_number, (int, np.integer)) or harmonic_number < 1:
        raise ValueError("harmonic_number must be a positive integer")
    if not np.isfinite(decay_rate) or decay_rate < 0:
        raise ValueError("decay_rate must be nonnegative and finite")

    # Initial amplitude and fade
    initial_weight = 1.0 / harmonic_number
    fade = np.exp(-decay_rate * harmonic_number * times)

    # Apply initial weight and fade
    return samples * initial_weight * fade
