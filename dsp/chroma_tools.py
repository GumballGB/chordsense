# Fold spectrum peaks into a 12-bin chroma vector, weighted by amplitude.
# Do note that chroma works only for chord detections, not so for individual note identification
# This is because chroma collapses all octaves into a single 12-bin representation, losing individual note information.

import numpy as np

from dsp.note_tools import frequency_to_note

# This function is to be used after finding peaks in the spectrum (frequency spectrum)
def build_chroma(peaks, a4_hz=440.0):
    """
    Sum peak amplitudes into 12 pitch-class bins, C through B.

    peaks is a list of dicts with "frequency_hz" and "amplitude", as returned
    by find_spectrum_peaks.
    a4_hz is the tuning reference passed on to frequency_to_note.

    Unlike a plain set of notes, every peak keeps its weight. Octave
    harmonics land in their own note's bin and make it stronger, while a
    faint non-octave harmonic only adds a little to a foreign bin.

    At this stage, the chroma vector represents 
    the summed amplitudes for each pitch class, ignoring octaves.

    Returns:
    numpy.ndarray: Shape (12,), index 0 is C and index 11 is B.
    """

    # Our 12-bin chroma vector, initialized to zero
    chroma = np.zeros(12)

    for p in peaks:

        # Extract the amplitude and validate it
        amplitude = float(p["amplitude"])
        if not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError("peak amplitudes must be nonnegative and finite")

        # MIDI number modulo 12 is the pitch class, octave is thrown away
        pitch_class = frequency_to_note(p["frequency_hz"], a4_hz=a4_hz)["midi"] % 12

        chroma[pitch_class] += amplitude

    # Return the completed chroma vector
    return chroma


def filter_chroma(chroma, threshold=0.0):
    """
    Keep the pitch classes whose bin is at least threshold times the strongest.

    chroma is a 12-bin vector from build_chroma.
    threshold is a fraction between 0 and 1 of the strongest bin.

    The threshold is now legacy, it has been kept for compatibility.
    All bins are considered, the cosine similarity is now taken into account
    instead of a hard threshold. Regardless, the threshold is kept for now.

    Returns:
    list[int]: Surviving pitch classes in ascending order, or [] if silent.
    """

    # Initialize a numpy array
    chroma = np.asarray(chroma, dtype=float)

    # Basic checks
    if chroma.shape != (12,):
        raise ValueError("chroma must have exactly 12 bins")
    if not np.all(np.isfinite(chroma)) or np.any(chroma < 0):
        raise ValueError("chroma must contain only nonnegative finite values")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    # Find the amplitude of the strongest bin
    # It will be used to determine which bins meet the threshold
    strongest = chroma.max()

    # Silence has no pitch classes
    if strongest == 0:
        return []

    # Empty bins never count, even with a threshold of 0
    keep = (chroma >= threshold * strongest) & (chroma > 0)
    
    # Return the pitch classes that meet the threshold
    return [int(pc) for pc in np.flatnonzero(keep)]
