# Tool function for detecting chords, turns MIDI note numbers into chord names.

# LOOK UP TABLES

TONES_LOOKUP = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B'
}

# All chords actually more or less have the same intervals, just shifted
# Actually, the root shifts!
CHORDS_LOOKUP = {
    (0, 4, 7): '',
    (0, 3, 7): 'm',
    (0, 4, 7, 11): 'maj7',
    (0, 3, 7, 10): 'm7',
    (0, 4, 7, 10): '7',
    (0, 3, 6): 'dim',
    (0, 3, 7, 11): 'mmaj7',

    # Power chords and suspended fourths
    (0, 7): '5',
    (0, 5, 7): 'sus4'
}

# The main function for detecting chords
def chord_extract(notes):
    """
    Match pitch classes against all roots, ignoring note order and doubling.

    Return a chord name, or None for unrecognized or empty inputs.
    The first matching root in C-to-B order wins.
    """
    normalized_notes = [n % 12 for n in notes]

    if not normalized_notes:
        return None

    pitch_classes = set(normalized_notes)

    # The first note may be an inversion, so try every possible root.
    for root in range(12):
        intervals = tuple(sorted((n - root) % 12 for n in pitch_classes))
        if intervals in CHORDS_LOOKUP:
            return TONES_LOOKUP[root] + CHORDS_LOOKUP[intervals]

    return None


