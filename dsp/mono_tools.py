# Convert audio channels into a single mono signal.

import numpy as np


def to_mono(signal):
    """
    Accept mono (samples,) or multichannel (samples, channels) audio.

    Multichannel audio is averaged across channels. Conversion to float happens
    first so adding integer audio samples cannot overflow.
    """
    samples = np.asarray(signal, dtype=float)

    if samples.ndim not in (1, 2):
        raise ValueError("signal must have shape (samples,) or (samples, channels)")
    if not np.all(np.isfinite(samples)):
        raise ValueError("signal must contain only finite samples")
    if samples.ndim == 1:
        return samples
    if samples.shape[1] == 0:
        raise ValueError("signal must have at least one channel")
    
    return samples.mean(axis=1)
