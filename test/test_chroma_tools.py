# Test the chroma stage, and that it fixes chords the plain pipeline misses

import numpy as np
import pytest

from dsp.chroma_tools import build_chroma, filter_chroma
from dsp.sine import harmonic_note, sine
from pipeline.chord_guess import (
    guess_chord,
    guess_chord_chroma,
    prepare_frame,
    signal_to_peaks,
)

C4, E4, G4 = 261.625565, 329.627557, 391.995436


def peak(frequency_hz, amplitude):
    return {'frequency_hz': frequency_hz, 'amplitude': amplitude}


def test_octaves_add_into_the_same_bin():
    chroma = build_chroma([peak(C4, 0.5), peak(2 * C4, 0.25), peak(E4, 0.3)])

    assert chroma.shape == (12,)
    assert chroma[0] == pytest.approx(0.75)
    assert chroma[4] == pytest.approx(0.3)
    assert np.count_nonzero(chroma) == 2


def test_empty_peaks_give_a_silent_chroma():
    np.testing.assert_array_equal(build_chroma([]), np.zeros(12))
    assert filter_chroma(np.zeros(12)) == []


def test_tuning_reference_is_passed_to_note_conversion():
    # 0.6 semitones above 432 Hz rounds to A# with A4 = 432,
    # but is only 0.28 semitones above A with A4 = 440.
    frequency = 432 * 2 ** (0.6 / 12)

    assert build_chroma([peak(frequency, 1.0)], a4_hz=432)[10] == 1.0
    assert build_chroma([peak(frequency, 1.0)])[9] == 1.0


def test_filter_keeps_bins_relative_to_the_strongest():
    chroma = np.zeros(12)
    chroma[[0, 4, 7, 11]] = [1.0, 0.9, 0.6, 0.59]

    assert filter_chroma(chroma, threshold=0.6) == [0, 4, 7]
    assert filter_chroma(chroma, threshold=0.0) == [0, 4, 7, 11]
    assert filter_chroma(chroma, threshold=1.0) == [0]


@pytest.mark.parametrize('chroma', [np.zeros(11), -np.ones(12), np.full(12, np.nan)])
def test_invalid_chroma(chroma):
    with pytest.raises(ValueError):
        filter_chroma(chroma)


@pytest.mark.parametrize('threshold', [-0.1, 1.1])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        filter_chroma(np.ones(12), threshold=threshold)


def test_invalid_amplitude():
    with pytest.raises(ValueError):
        build_chroma([peak(C4, -1.0)])


def test_weak_harmonic_is_ignored_by_chroma_but_not_by_guess_chord():
    # A faint B from E's third harmonic turns C major into Cmaj7
    # when every peak counts equally.
    peaks = [peak(C4, 0.3), peak(E4, 0.3), peak(G4, 0.3), peak(3 * E4, 0.1)]

    assert guess_chord(peaks) == 'Cmaj7'
    assert guess_chord_chroma(peaks) == 'C'


def harmonic_c_major(**kwargs):
    chord = sum(harmonic_note(f, 1, **kwargs) for f in (C4, E4, G4))
    return chord / np.max(np.abs(chord))


@pytest.mark.parametrize(('kwargs', 'expected'), [
    ({'damped': True}, 'C'),
    ({'damped': True, 'decay_rate': 0.5}, 'C'),
    ({}, 'Cmaj7'),
])
def test_harmonic_c_major_template_results(kwargs, expected):
    # Equal-strength upper harmonics make Cmaj7 the closest binary template
    # in the undamped case, even though C major was played.
    frame, weights = prepare_frame(harmonic_c_major(**kwargs))
    peaks = signal_to_peaks(frame, window_weights=weights)

    assert guess_chord_chroma(peaks) == expected


@pytest.mark.parametrize(('midi_notes', 'expected'), [
    ([60, 64, 67], 'C'),
    ([52, 55, 60], 'C'),
    ([57, 60, 64, 67], 'Am7'),
    ([40, 47, 52, 59, 64], 'E5'),
])
def test_pure_sine_chords_still_work_with_chroma(midi_notes, expected):
    frequencies = [440 * 2 ** ((n - 69) / 12) for n in midi_notes]
    signal = sum(sine(f, 1) for f in frequencies) / len(frequencies)

    frame, weights = prepare_frame(signal)
    peaks = signal_to_peaks(frame, window_weights=weights)

    assert guess_chord_chroma(peaks) == expected
