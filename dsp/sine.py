# default sine wave
import numpy as np

from dsp.harmonic_tools import dampen_harmonics

def sine(frequency, duration, sample_rate=44100):
    """
    Generate a sine wave of a given frequency and duration.

    Parameters:
    frequency (float): Frequency of the sine wave in Hz.
    duration (float): Duration of the sine wave in seconds.
    sample_rate (int): Sample rate in samples per second. Default is 44100.

    Returns:
    numpy.ndarray: Array containing the sine wave samples.
    """

    t = np.linspace(0, duration, round(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t)

def cosine(frequency, duration, sample_rate=44100):
    """
    Generate a cosine wave of a given frequency and duration.

    Parameters:
    frequency (float): Frequency of the cosine wave in Hz.
    duration (float): Duration of the cosine wave in seconds.
    sample_rate (int): Sample rate in samples per second. Default is 44100.

    Returns:
    numpy.ndarray: Array containing the cosine wave samples.
    """

    t = np.linspace(0, duration, round(sample_rate * duration), endpoint=False)
    return np.cos(2 * np.pi * frequency * t)

def harmonic_note(
    frequency, duration, sample_rate=44100, n_harmonics=5, *, damped=False, decay_rate=2.0
):
    """
    Add a fundamental and its harmonics, optionally shaping their timbre.

    frequency is the fundamental in Hz.
    duration is in seconds.
    sample_rate is in Hz.
    n_harmonics is the number of harmonics to include.
    damped (bool): Whether to apply harmonic damping. For musical notes, this can make them sound more natural.
    decay_rate (float): How fast higher harmonics fade.

    n_harmonics includes the fundamental. With damped=True, higher harmonics
    start quieter and fade faster. 
    
    Skip harmonics at or above Nyquist and scale
    the finished note down if needed to keep its samples within [-1, 1].
    """

    # The standard checks for valid input
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite")
    if not np.isfinite(frequency) or not 0 < frequency < sample_rate / 2:
        raise ValueError("frequency must be positive and below Nyquist")
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be nonnegative and finite")
    if not isinstance(n_harmonics, (int, np.integer)) or n_harmonics < 1:
        raise ValueError("n_harmonics must be a positive integer")

    # Sample times in seconds: 0, 1/sample_rate, 2/sample_rate, ...
    # If at 44100Hz, seconds are like 0, 1/44100, 2/44100, ...
    n_samples = round(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # One zero amplitude per sample time, ready to accumulate the harmonics.
    wave_amplitude = np.zeros_like(t)

    for i in range(1, n_harmonics + 1):

        # Current harmonic number is i, harmonics are multiples
        harmonic_frequency = frequency * i

        # Break is above Nyquist
        if harmonic_frequency >= sample_rate / 2:
            break

        # Make the harmonic wave
        harmonic = np.sin(2 * np.pi * harmonic_frequency * t)

        # If damped, shape the harmonic's timbre, by changing amplitude over time
        if damped:
            harmonic = dampen_harmonics(harmonic, t, i, decay_rate=decay_rate)

        wave_amplitude += harmonic

    # Normalize if peak exceeds 1
    peak = np.max(np.abs(wave_amplitude), initial=0.0)
    
    if peak > 1:
        wave_amplitude = wave_amplitude / peak

    return wave_amplitude
