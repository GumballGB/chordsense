import math

NOTE_NAMES_SHARP = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]


def frequency_to_note(frequency_hz, a4_hz=440.0):
    """
    Map a frequency to the nearest equal-tempered note.

    Parameters:
    frequency_hz (float): Input frequency in Hz.
    a4_hz (float): Reference tuning frequency for A4.

    The central tuning for A4 is typically 440 Hz, 
    but this can be adjusted for different tuning systems.

    Returns:
    dict: {
        "note": str,
        "midi": int,
        "nearest_freq_hz": float,
        "delta_hz": float,
        "cents": float,
    }

    To further explain the output:
    - "note": The name of the nearest note (e.g., "A4").
    - "midi": The MIDI note number corresponding to the nearest note.
    - "nearest_freq_hz": The frequency of the nearest note in Hz.
    - "delta_hz": The difference between the input frequency and the nearest note's frequency.
    - "cents": The difference in cents between the input frequency and the nearest note's frequency. 
    A positive value indicates the input frequency is higher than the nearest note, while a negative value

    Example:
    frequency_to_note(444.0) -> {"note": "A4", "delta_hz": 4.0, ...}
    """

    # prevent invalid input values
    frequency_hz = float(frequency_hz)
    a4_hz = float(a4_hz)

    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0")
    if a4_hz <= 0:
        raise ValueError("a4_hz must be > 0")

    # Figure out the MIDI note number for the given frequency
    # 69 represents A4 in MIDI, which is the reference note for tuning.
    midi_float = 69.0 + 12.0 * math.log2(frequency_hz / a4_hz)

    # MIDI are integers, so we round to the nearest whole number to find the closest note.
    midi = int(round(midi_float))

    # Figure out the frequency of the nearest note and how far off we are in cents
    nearest_freq_hz = a4_hz * (2.0 ** ((midi - 69) / 12.0))
    cents = (midi_float - midi) * 100.0
    delta_hz = frequency_hz - nearest_freq_hz

    # Finally, we determine the note name and octave based on the MIDI number.
    note_name = NOTE_NAMES_SHARP[midi % 12]
    octave = (midi // 12) - 1

    return {
        "note": f"{note_name}{octave}",
        "midi": midi,
        "nearest_freq_hz": nearest_freq_hz,
        "delta_hz": delta_hz,
        "cents": cents,
    }
