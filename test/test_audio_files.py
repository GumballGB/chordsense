import numpy as np
import pytest
from scipy.io import wavfile

from audio.file_tools import load_audio, save_audio
from pipeline.chord_guess import prepare_frame, signal_to_peaks, guess_chord_chroma


@pytest.mark.parametrize('sample_rate', [44100, 48000])
@pytest.mark.parametrize('channels', [1, 2])
def test_float_recording_round_trip_preserves_samples_and_metadata(tmp_path, sample_rate, channels):
    # Quiet samples must not be amplified by saving or loading.
    samples = np.linspace(-0.01, 0.01, 100 * channels, dtype=np.float32).reshape(100, channels)
    path = save_audio(tmp_path / 'recordings' / 'quiet.wav', samples, sample_rate)
    loaded, rate = load_audio(path)
    assert rate == sample_rate
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, samples)


def test_one_dimensional_mono_loads_with_a_channel_axis(tmp_path):
    path = save_audio(tmp_path / 'mono.wav', np.array([0.0, 0.5, -0.5]), 44100)
    loaded, _ = load_audio(path)
    assert loaded.shape == (3, 1)
    np.testing.assert_array_equal(loaded[:, 0], [0.0, 0.5, -0.5])


@pytest.mark.parametrize(('pcm', 'expected'), [
    (np.array([-32768, 0, 16384], dtype=np.int16), [-1.0, 0.0, 0.5]),
    (np.array([0, 128, 192], dtype=np.uint8), [-1.0, 0.0, 0.5]),
    (np.array([-2147483648, 0, 1073741824], dtype=np.int32), [-1.0, 0.0, 0.5]),
])
def test_integer_pcm_is_converted_using_full_scale(tmp_path, pcm, expected):
    path = tmp_path / 'pcm.wav'
    wavfile.write(str(path), 16000, pcm)
    loaded, rate = load_audio(path)
    assert rate == 16000
    np.testing.assert_array_equal(loaded[:, 0], expected)


@pytest.mark.parametrize('samples', [
    np.array([]), np.zeros((2, 0)), np.zeros((2, 2, 2)),
    np.array([np.nan]), np.array([np.inf]), np.array([32767], dtype=np.int16),
])
def test_invalid_samples_are_not_saved(tmp_path, samples):
    path = tmp_path / 'invalid.wav'
    with pytest.raises(ValueError):
        save_audio(path, samples, 44100)
    assert not path.exists()


@pytest.mark.parametrize('rate', [0, -1, 44100.5, np.nan])
def test_invalid_sample_rate_is_not_saved(tmp_path, rate):
    with pytest.raises(ValueError):
        save_audio(tmp_path / 'invalid.wav', np.zeros(10), rate)


def test_reloaded_chord_still_matches_at_its_saved_sample_rate(tmp_path):
    rate = 48000
    times = np.arange(rate) / rate
    samples = sum(np.sin(2 * np.pi * f * times) for f in (261.625565, 329.627557, 391.995436)) / 3
    path = save_audio(tmp_path / 'c-major.wav', samples.astype(np.float32), rate)
    loaded, loaded_rate = load_audio(path)

    frame, weights = prepare_frame(loaded, sample_rate=loaded_rate, frame_size=loaded_rate)
    peaks = signal_to_peaks(frame, loaded_rate, window_weights=weights)
    assert guess_chord_chroma(peaks) == 'C'
