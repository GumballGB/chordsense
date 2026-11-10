# Test the chord guessing pipeline
# we use synthesized sine waves for testing

from copy import deepcopy

import numpy as np
import pytest

from dsp.sine import sine
from dsp.mono_tools import to_mono
from pipeline.chord_guess import guess_chord, prepare_frame, signal_to_peaks


@pytest.mark.parametrize(('midi_notes', 'expected'), [
    ([60, 64, 67], 'C'),
    ([52, 55, 60], 'C'),  # E3, G3, C4: inverted C major
    ([57, 60, 64, 67], 'Am7'),
    ([40, 47, 52, 59, 64], 'E5'),
    ([40, 47, 52, 57, 59, 64], 'Esus4'),
])
@pytest.mark.parametrize('sample_rate', [44100, 16000])
def test_synthesized_chord_pipeline(midi_notes, expected, sample_rate):
    frequencies = [440 * 2 ** ((n - 69) / 12) for n in midi_notes]
    signal = sum(sine(f, 1, sample_rate) for f in frequencies) / len(frequencies)

    frame, weights = prepare_frame(signal, frame_size=sample_rate)
    peaks = signal_to_peaks(frame, sample_rate=sample_rate, window_weights=weights)

    assert len(peaks) == len(set(midi_notes))
    assert guess_chord(peaks) == expected


@pytest.mark.parametrize('signal', [[], [0.0], np.zeros(44100), np.ones(44100)])
def test_empty_silent_or_constant_signal_has_no_chord(signal):
    peaks = signal_to_peaks(signal)
    assert peaks == []
    assert guess_chord(peaks) is None


def test_single_a4_is_a_note_but_not_a_chord():
    peaks = signal_to_peaks(sine(440, 1))
    assert len(peaks) == 1
    assert peaks[0]['frequency_hz'] == pytest.approx(440)
    assert guess_chord(peaks) is None


def test_stereo_requires_mono_conversion():
    with pytest.raises(ValueError, match='mono'):
        signal_to_peaks(np.zeros((100, 2)))


def test_pipeline_does_not_implicitly_crop_to_one_second():
    chord = sum(sine(f, 1) for f in (261.625565, 329.627557, 391.995436)) / 3
    signal = np.concatenate([np.zeros(44100), chord])

    assert guess_chord(signal_to_peaks(signal)) == 'C'


def test_stereo_chord_through_preparation_and_detection():
    chord = sum(sine(f, 1) for f in (261.625565, 329.627557, 391.995436)) / 3
    stereo = np.column_stack([chord, 0.5 * chord])

    frame, weights = prepare_frame(stereo)

    assert frame.shape == (44100,)
    assert frame[0] == frame[-1] == 0
    assert guess_chord(signal_to_peaks(frame, window_weights=weights)) == 'C'


def test_mono_conversion_averages_channels_without_integer_overflow():
    stereo = np.array([[30000, 30000], [-30000, 10000]], dtype=np.int16)
    np.testing.assert_array_equal(to_mono(stereo), [30000, -10000])


def test_preparation_takes_only_the_requested_frame():
    c_major = sum(sine(f, 1) for f in (261.625565, 329.627557, 391.995436)) / 3
    a_minor = sum(sine(f, 1) for f in (220, 261.625565, 329.627557)) / 3
    signal = np.concatenate([c_major, a_minor])

    frame, weights = prepare_frame(signal, frame_size=44100)

    assert frame.shape == (44100,)
    assert guess_chord(signal_to_peaks(frame, window_weights=weights)) == 'C'


@pytest.mark.parametrize('signal', [[], np.empty((0, 2)), np.full((44100, 2), 1000)])
def test_preparation_handles_empty_or_constant_audio(signal):
    frame, weights = prepare_frame(signal)
    peaks = signal_to_peaks(frame, window_weights=weights)
    assert peaks == []
    assert guess_chord(peaks) is None


@pytest.mark.parametrize('sample_rate', [16000, 16001])
def test_windowed_peak_reports_original_tone_amplitude(sample_rate):
    # An integer-Hz tone in a one-second frame lands exactly on an FFT bin.
    signal = 0.33 * sine(440, 1, sample_rate) + 0.25
    frame, weights = prepare_frame(signal, frame_size=sample_rate)
    peaks = signal_to_peaks(frame, sample_rate=sample_rate, window_weights=weights)

    assert len(peaks) == 1
    assert peaks[0]['frequency_hz'] == pytest.approx(440)
    assert peaks[0]['amplitude'] == pytest.approx(0.33, rel=1e-5)
    # Preparation removes the offset once, before tapering the waveform.
    np.testing.assert_allclose(frame, 0.33 * sine(440, 1, sample_rate) * weights, atol=1e-14)


def test_guess_chord_preserves_peak_dictionaries():
    peaks = [
        {'frequency_hz': 261.625565, 'amplitude': 0.33, 'midi': 'caller metadata'},
        {'frequency_hz': 329.627557, 'amplitude': 0.33},
        {'frequency_hz': 391.995436, 'amplitude': 0.33},
    ]
    original = deepcopy(peaks)

    assert guess_chord(peaks) == 'C'
    assert peaks == original
