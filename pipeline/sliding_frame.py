# File for sliding frame processing of audio

import numpy as np
import pipeline.chord_guess as chord_guess
import dsp.chroma_tools as chroma_tools
import dsp.template_tools as template_tools
import dsp.mono_tools as mono


def analyze_frame(frame, sample_rate, min_score=0.8):
    """
    Given a frame of audio, analyze it for chords.

    Returns the detected chord for the frame, along with the score and chroma.

    This is an alternative of the pipeline which does not consider just one frame.
    """

    outputs = chord_guess.full_pipeline(frame, sample_rate, min_score=min_score, find_start=False)
    
    return outputs

    


def chords_over_time(signal, sample_rate, frame_seconds=0.5, hop_seconds=0.1):
    """
    Slide a window over the audio and detect a chord in each frame.

    signal: input audio signal, mono or stereo.
    frame_seconds: length of each analysis window. Longer frames give better
    frequency resolution but blur changes in time.

    hop_seconds: how far the window moves between guesses. Shorter hops
    track changes more precisely.

    Example with frame=1.0 s, hop=0.1 s:
    analyze 0.0-1.0 s, 
    then 0.1-1.1 s, 
    then 0.2-1.2 s, ...

    Returns a list of dicts: {"time": frame centre in seconds,
    "chord": detected chord, "score": confidence score}
    """
    signal = mono.to_mono(signal)

    # Convert frame length to samples
    frame_samples = round(frame_seconds * sample_rate)
    hop_samples = round(hop_seconds * sample_rate)

    # Initialize results list
    results = []

    # Slide over the signal 
    for start in range(0, len(signal) - frame_samples + 1, hop_samples):

        # Extract the current frame
        frame = signal[start:start + frame_samples]

        # Analyze this frame
        analysis = analyze_frame(frame, sample_rate)

        # Store the result for this frame
        results.append({
            "time": (start + frame_samples / 2) / sample_rate,
            "chord": analysis["chord"],
            "score": analysis["score"],
        })

    return results