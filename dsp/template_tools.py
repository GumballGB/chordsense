# Score weighted chroma against the chord vocabulary using cosine similarity.

import numpy as np

from dsp.chord_tools import CHORDS_LOOKUP, TONES_LOOKUP


def cosine_similarity(a, b):
    """
    Return the cosine score of two finite, equally sized 1D vectors.

    A zero vector has no direction, so return 0 when either input is zero.
    A score measures similarity, not the probability of a correct prediction.
    """

    # Convert to float arrays for dot product.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Zero vector check
    if a.ndim != 1 or a.size == 0 or a.shape != b.shape:
        raise ValueError("vectors must be nonempty matching 1D arrays")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("vectors must contain only finite values")

    # Rescale first so very quiet or loud inputs cannot underflow or overflow.
    a_scale = np.max(np.abs(a))
    b_scale = np.max(np.abs(b))
    if a_scale == 0 or b_scale == 0:
        return 0.0

    # Normalize vectors before dot product.
    a = a / a_scale
    b = b / b_scale

    # Compute cosine similarity.
    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Clip score to valid range.
    return float(np.clip(score, -1.0, 1.0))


def build_templates():
    """
    Return name/vector pairs for every root and shape in the vocabulary.

    Each template has 12 bins: 1 for a chord tone, 0 otherwise. Chroma itself
    keeps its original weights. There are currently 12 roots x 9 shapes.

    For example the C major chord has 1s at C, E, G. So it's [1,0,0,0,1,0,0,1,0,0,0,0]
    The E minor would have 1s at E, G, B. So it's [0,0,1,0,0,0,1,0,0,1,0,0]

    As said previously, it returns all root/shape combinations. There are 12 roots x 9 shapes.
    """

    # Build all chord templates.
    templates = {}

    # Loop over roots and shapes to create templates.
    for root in range(12):

        # Loop over each chord shape for this root.
        for intervals, suffix in CHORDS_LOOKUP.items():

            # Make template for this chord.
            template = np.zeros(12)

            # Fill in chord tones.
            for interval in intervals:
                template[(root + interval) % 12] = 1.0

            # Save template with name.
            name = TONES_LOOKUP[root] + suffix
            templates[name] = template

    return templates


def match_chord(chroma):
    """
    Return (best_name, score), or (None, 0.0) for silent chroma.

    Keep every chroma weight and compare every template. Exact ties keep the
    first candidate in C-to-B root order, then CHORDS_LOOKUP insertion order.
    No acceptance cutoff is applied here; the caller can inspect the score.
    """

    # Convert chroma to float array.
    chroma = np.asarray(chroma, dtype=float)

    # Check shape and values.
    if chroma.shape != (12,):
        raise ValueError("chroma must have exactly 12 bins")
    if not np.all(np.isfinite(chroma)) or np.any(chroma < 0):
        raise ValueError("chroma must contain only nonnegative finite values")
    if not np.any(chroma):
        return None, 0.0

    # Initialize best match.
    best_name = None
    best_score = 0.0

    # Compare against all templates.
    for name, template in build_templates().items():

        score = cosine_similarity(chroma, template)

        # Update best if needed.
        if score > best_score:
            best_name = name
            best_score = score

    return best_name, best_score
