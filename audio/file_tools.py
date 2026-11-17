"""Save recordings and reload the same samples for repeatable experiments."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def _float_samples(samples):
    """Validate floating-point audio and keep a channel axis for mono."""

    # Convert to array and check shape
    samples = np.asarray(samples)

    # Must be nonempty
    if samples.ndim not in (1, 2) or samples.size == 0:
        raise ValueError('samples must be nonempty mono or multichannel audio')
    if not np.issubdtype(samples.dtype, np.floating):
        raise ValueError('samples must be floating-point audio, as returned by record_audio')

    # Convert to float32
    samples = samples.astype(np.float32)

    # Check for finite values first
    if not np.all(np.isfinite(samples)):
        raise ValueError('samples must contain only finite values')
    if samples.ndim == 1:
        samples = samples[:, None]

    return samples


def save_audio(path, samples, sample_rate):
    """
    Save raw audio as an uncompressed float32 WAV and return its Path.

    Preserve amplitude and channel layout; do not peak-normalize or window the
    recording. Float32 microphone samples are preserved exactly. Parent folders
    are created as needed; saving to the same path replaces the previous file.
    """

    # Check sample rate first
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError('sample_rate must be a positive integer')

    # Validate samples
    samples = _float_samples(samples)

    # Make sure parent folder exists
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), int(sample_rate), samples)

    # Write WAV file
    return path


def load_audio(path):
    """
    Return (float32_samples, sample_rate), matching record_audio's format.

    Mono audio has shape (N, 1). Floating WAV amplitudes are left unchanged.
    Integer PCM is converted using the format's full-scale range, not the
    recording's loudest sample, so quiet recordings stay quiet.
    """

    # Read WAV file
    sample_rate, samples = wavfile.read(str(path))

    # Convert integer PCM to float32
    if samples.dtype == np.uint8:
        # 8-bit PCM is unsigned: 128 represents silence.
        samples = (samples.astype(np.float32) - 128) / 128

    # Convert signed integers to float32
    elif np.issubdtype(samples.dtype, np.signedinteger):
        # SciPy returns 24-bit PCM left-justified in int32, so this also
        # handles it correctly using the int32 full-scale range.
        full_scale = float(2 ** (np.iinfo(samples.dtype).bits - 1))
        samples = samples.astype(np.float32) / full_scale

    # Convert float WAV to float32
    return _float_samples(samples), int(sample_rate)
