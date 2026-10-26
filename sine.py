# default sine wave
import numpy as np

def sine(frequency, duration, sample_rate=44100):
    """
    Generate a sine wave of a given frequency and duration.

    Parameters:
    freq (float): Frequency of the sine wave in Hz.
    duration (float): Duration of the sine wave in seconds.
    sample_rate (int): Sample rate in samples per second. Default is 44100.

    Returns:
    numpy.ndarray: Array containing the sine wave samples.
    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t)

def cosine(frequency, duration, sample_rate=44100):
    """
    Generate a cosine wave of a given frequency and duration.

    Parameters:
    freq (float): Frequency of the cosine wave in Hz.
    duration (float): Duration of the cosine wave in seconds.
    sample_rate (int): Sample rate in samples per second. Default is 44100.

    Returns:
    numpy.ndarray: Array containing the cosine wave samples.
    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.cos(2 * np.pi * frequency * t)