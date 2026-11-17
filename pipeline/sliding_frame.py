# File for sliding frame processing of audio

import numpy as np
import pipeline.chord_guess as chord_guess
import dsp.chroma_tools as chroma_tools
import dsp.template_tools as template_tools
import dsp.mono_tools as mono
from collections import Counter


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
            "time": (start + frame_samples / 2) / sample_rate, # becomes important for segments
            "chord": analysis["chord"],
            "score": analysis["score"],
        })

    return results

# Function for smoothing results
def smooth_results(results, window=5):
    """
    Smooth the chord sequence using a majority vote in a window.

    results is a list of dicts with "chord" keys.
    window: number of frames to consider for smoothing. It must be odd
    """

    # Initialize smoothed list
    smoothed = []
    # Loop over each result and smooth it
    for i in range(len(results)):

        # Start of the window
        start = max(0, i - window//2)
        # End of the window
        end = min(len(results), i + window//2 + 1)

        # Get chords in the window
        chords = [r["chord"] for r in results[start:end]]

        # Majority vote. If there is a tie, just pick the first one
        chord = Counter(chords).most_common(1)[0][0]

        # Save smoothed chord
        smoothed.append({**results[i], "chord": chord})

    return smoothed

def to_segments(results, hop_seconds):
    """
    Convert frame results to segments. Making it more clean
    results is a list of dicts with "chord" keys.
    Usually results are smoothed before this.
    Usually called by chords_over_time, since its results include "time" for each frame.

    result is a list of segments.
    """
    # Initialize segments list
    segments = []

    # If no results, return empty list
    if not results:
        return segments

    start = 0

    # Loop over results to make segments
    for i in range(1, len(results)):

        # Check for chord change, note that every result has the correct centre time.
        # Therefore to get the right time, use the centre of the frame offset by hop_seconds
        if results[i]["chord"] != results[start]["chord"]:

            segments.append({"chord": results[start]["chord"], "start": results[start]["time"] - hop_seconds/2, "end": results[i-1]["time"] + hop_seconds/2})
            # Start new segment, the start is now i 
            start = i
        # Otherwise keep going if chord is same
    
    # Add last segment
    segments.append({"chord": results[start]["chord"], "start": results[start]["time"] - hop_seconds/2, "end": results[-1]["time"] + hop_seconds/2})
    
    # Return all segments
    return segments