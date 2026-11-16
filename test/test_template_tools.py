import numpy as np
import pytest

from dsp.template_tools import build_templates, cosine_similarity, match_chord
from pipeline.chord_guess import guess_chord_chroma


def test_notebook_cosine_example():
    chroma = np.zeros(12)
    chroma[[0, 4, 7]] = [1.0, 1.0, 0.5]
    template = build_templates()['C']

    assert cosine_similarity(chroma, template) == pytest.approx(0.9622504486)
    assert match_chord(chroma)[0] == 'C'


@pytest.mark.parametrize('scale', [1e-200, 1.0, 1e200])
def test_similarity_is_independent_of_volume(scale):
    assert cosine_similarity(np.array([1.0, 1.0, 0.5]) * scale, [1, 1, 1]) == pytest.approx(0.9622504486)


def test_zero_orthogonal_and_opposite_vectors():
    assert cosine_similarity([0, 0], [1, 1]) == 0
    assert cosine_similarity([1, 0], [0, 1]) == 0
    assert cosine_similarity([1, 0], [-1, 0]) == -1


@pytest.mark.parametrize(('a', 'b'), [
    ([], []), ([1], [1, 2]), ([[1]], [[1]]), ([np.nan], [1]), ([1], [np.inf]),
])
def test_invalid_cosine_inputs(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_all_roots_and_shapes_are_built():
    templates = build_templates()
    assert len(templates) == 108
    assert all(t.shape == (12,) and set(t) <= {0, 1} for t in templates.values())
    np.testing.assert_array_equal(np.flatnonzero(templates['B7']), [3, 6, 9, 11])
    np.testing.assert_array_equal(np.flatnonzero(templates['Fsus4']), [0, 5, 10])


@pytest.mark.parametrize(('pitch_classes', 'expected'), [
    ([0, 4, 7], 'C'),
    ([0, 4, 9], 'Am'),
    ([1, 2, 6, 9], 'Dmaj7'),
    ([0, 3, 5, 8], 'Fm7'),
    ([3, 6, 9, 11], 'B7'),
    ([1, 4, 7], 'C#dim'),
    ([2, 3, 6, 10], 'D#mmaj7'),
    ([4, 11], 'E5'),
    ([0, 5, 10], 'Fsus4'),
])
def test_every_shape_can_be_an_exact_template_match(pitch_classes, expected):
    chroma = np.zeros(12)
    chroma[pitch_classes] = 1
    name, score = match_chord(chroma)
    assert name == expected
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize('chroma', [np.zeros(11), -np.ones(12), np.full(12, np.nan)])
def test_invalid_chroma(chroma):
    with pytest.raises(ValueError):
        match_chord(chroma)


def test_silence_has_no_match_even_without_a_score_cutoff():
    assert match_chord(np.zeros(12)) == (None, 0.0)
    assert guess_chord_chroma([], min_score=0) is None


def test_single_note_and_flat_noise_are_rejected_by_default():
    assert guess_chord_chroma([{'frequency_hz': 261.625565, 'amplitude': 1}]) is None
    peaks = [{'frequency_hz': 440 * 2 ** ((60 + pc - 69) / 12), 'amplitude': 1} for pc in range(12)]
    assert guess_chord_chroma(peaks) is None
    assert match_chord(np.ones(12))[1] == pytest.approx(np.sqrt(4 / 12))


def test_weak_chord_tone_is_scored_instead_of_removed():
    peaks = [
        {'frequency_hz': 261.625565, 'amplitude': 1.0},
        {'frequency_hz': 329.627557, 'amplitude': 1.0},
        {'frequency_hz': 391.995436, 'amplitude': 0.5},
    ]
    assert guess_chord_chroma(peaks) == 'C'
    assert guess_chord_chroma(peaks, min_score=0.97) is None
    assert guess_chord_chroma(peaks, min_score=0.95) == 'C'


@pytest.mark.parametrize('min_score', [-0.1, 1.1, np.nan, np.inf])
def test_invalid_acceptance_cutoff(min_score):
    with pytest.raises(ValueError):
        guess_chord_chroma([], min_score=min_score)


def test_uneven_voicing_can_still_prefer_a_power_chord():
    chroma = np.zeros(12)
    chroma[[4, 9, 11]] = [3, 1, 2]  # E, A, B in an Esus4 guitar voicing
    name, score = match_chord(chroma)
    sus_score = cosine_similarity(chroma, build_templates()['Esus4'])
    assert name == 'E5'
    assert score > sus_score > 0.9
