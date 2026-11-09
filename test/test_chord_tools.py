# Test file for testing if chords extraction works.

import pytest

from dsp.chord_tools import chord_extract


def test_chord():
    assert chord_extract([60, 64, 67]) == 'C'
    assert chord_extract([60, 63, 67]) == 'Cm'
    assert chord_extract([60, 64, 67, 71]) == 'Cmaj7'
    assert chord_extract([60, 63, 66]) == 'Cdim'
    assert chord_extract([60, 64, 67, 70]) == 'C7'
    assert chord_extract([60, 63, 67, 70]) == 'Cm7'
    assert chord_extract([60, 64, 68]) is None
    assert chord_extract([60, 63, 67, 71]) == 'Cmmaj7'
    assert chord_extract([60, 64, 67, 69]) == 'Am7'  # C, E, G, A
    assert chord_extract([60, 67, 71]) is None
    assert chord_extract([60, 64, 67, 72]) == 'C'


@pytest.mark.parametrize(('notes', 'expected'), [
    ([52, 55, 60], 'C'),  # E3, G3, C4: first inversion
    ([55, 60, 64], 'C'),  # G3, C4, E4: second inversion
    ([63, 67, 72], 'Cm'),
    ([64, 67, 71, 72], 'Cmaj7'),
    ([63, 66, 72], 'Cdim'),
    ([64, 67, 70, 72], 'C7'),
    ([63, 67, 70, 72], 'Cm7'),
    ([63, 67, 71, 72], 'Cmmaj7'),
    ([67, 64, 60], 'C'),  # Unsorted input
    ([76, 55, 60, 64, 72, 67, 60], 'C'),  # Repeated notes and octaves
    ([40, 47, 52, 59, 64], 'E5'),  # Guitar voicing with octave doubling
    ([40, 47, 52, 57, 59, 64], 'Esus4'),
])
def test_chord_voicings(notes, expected):
    assert chord_extract(notes) == expected


@pytest.mark.parametrize(('root', 'name'), list(enumerate([
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
])))
def test_major_inversions_in_every_key(root, name):
    for notes in ([60, 64, 67], [64, 67, 72], [67, 72, 76]):
        assert chord_extract([n + root for n in notes]) == name


@pytest.mark.parametrize('notes', [
    [],
    [60],
    [60, 64],  # A major-third dyad is not sus4
    [60, 67, 71],
    [60, 64, 68],
    [68, 60, 64],  # Unknown stays None regardless of input order
    [64, 68, 72],
])
def test_unrecognized_chord_returns_none(notes):
    assert chord_extract(notes) is None


@pytest.mark.parametrize(('root', 'name'), list(enumerate([
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
])))
@pytest.mark.parametrize(('notes', 'suffix'), [
    ([60, 67], '5'),
    ([60, 65, 67], 'sus4'),
])
def test_guitar_chords_in_every_key(root, name, notes, suffix):

    transposed = [n + root for n in notes]
    
    for inversion in range(len(transposed)):
        voicing = transposed[inversion:] + [n + 12 for n in transposed[:inversion]]
        assert chord_extract(voicing) == name + suffix
        assert chord_extract(list(reversed(voicing)) + [voicing[0] + 12]) == name + suffix
