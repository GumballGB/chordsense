"""
Record a short microphone clip as samples for the existing DSP pipeline.

The microphone/audio interface performs analog-to-digital conversion. This
module asks sounddevice to deliver those digital samples as a NumPy array.
Importing it does not start recording; call record_audio when ready.

sounddevice is used for recording and it is supposed to work on mac and windows.
Linux requires an additional import (lipportaudio2) for sounddevice.
"""

import numpy as np
import sounddevice as sd


def list_input_devices():
    """
    Return sounddevice device dictionaries for microphones/audio inputs.

    The sd library has a function query_devices to list them.
    """

    return [device for device in sd.query_devices() if device['max_input_channels'] > 0]


def record_audio(duration=1.0, sample_rate=None, channels=1, device=None):
    """
    Return (samples, sample_rate) after a fixed-duration recording finishes.


    duration is in seconds. device is a sounddevice input ID/name, or None for
    the default input. 
    
    sample_rate defaults to that device's preferred rate. For example, the macbook pro's
    sample rate is 48000.

    Samples are float32 with shape (sample_count, channels), even for mono.
    Pass them to prepare_frame for mono conversion and Hann windowing.
    """

    # Validate duration and channels first
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be positive and finite')
    if not isinstance(channels, (int, np.integer)) or channels < 1:
        raise ValueError('channels must be a positive integer')

    # Validate sample rate next, sample rates can vary from device to device
    if sample_rate is None:
        info = sd.query_devices(device, kind='input')
        sample_rate = info['default_samplerate']

    # Make sure sample rate is valid
    if not np.isfinite(sample_rate) or sample_rate <= 0 or int(sample_rate) != sample_rate:
        raise ValueError('sample_rate must be a positive whole number of Hz')

    # At this point, sample_rate is valid and finite.
    # Convert the sample rate to int for consistency
    sample_rate = int(sample_rate)

    # Compute sample count
    sample_count = round(duration * sample_rate)
    if sample_count < 1:
        raise ValueError('duration is too short to record one sample')

    # Reject unsupported settings before opening a recording stream.
    sd.check_input_settings(
        device=device, samplerate=sample_rate, channels=channels, dtype='float32'
    )

    # Finally, start recording
    samples = sd.rec(
        sample_count,
        samplerate=sample_rate,
        channels=channels,
        dtype='float32',
        device=device,
        blocking=True,  # Wait until every requested sample has arrived.
    )

    # Check for overflow after recording
    # Overflow means some samples were lost
    if sd.get_status().input_overflow:
        raise RuntimeError('Audio input overflowed; some samples were lost. Try recording again.')

    # Return the recorded samples and rate
    return samples, sample_rate
