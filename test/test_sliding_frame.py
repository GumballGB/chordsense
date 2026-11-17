# Test for sliding frame chord analysis

import pipeline.sliding_frame as sliding_frame
import dsp.sine as sine
import pytest
import numpy as np

def test_chords_over_time():
    
    signal = [0]*44100
    result = sliding_frame.chords_over_time(signal, 44100)
    assert isinstance(result, list)
    
    if result:
        assert "time" in result[0]
        assert "chord" in result[0]
        assert "score" in result[0]

def test_empty_signal():
    result = sliding_frame.chords_over_time([], 44100)
    assert result == []

def test_short_signal():
    result = sliding_frame.chords_over_time([0]*100, 44100)
    assert result == []

def test_hop():
    result = sliding_frame.chords_over_time([0]*44100, 44100, hop_seconds=0.2)
    assert isinstance(result, list)
    if result:
        assert "time" in result[0]
        assert "chord" in result[0]
        assert "score" in result[0]

def test_hop_short():
    result = sliding_frame.chords_over_time([0]*100, 44100, hop_seconds=0.2)
    assert result == []

# Test the frame count, 2s of audio, 0.5s of frame, 0.1s hop
def test_frame_count():
    result = sliding_frame.chords_over_time([0]*88200, 44100, frame_seconds=0.5, hop_seconds=0.1)
    assert len(result) == 16


def test_no_cropping():
    result = sliding_frame.chords_over_time([0]*44100, 44100, frame_seconds=0.5, hop_seconds=0.1)
    assert isinstance(result, list)
    if result:
        assert "time" in result[0]
        assert "chord" in result[0]
        assert "score" in result[0]


def test_centre_time():
    result = sliding_frame.chords_over_time([0]*44100, 44100, frame_seconds=0.5, hop_seconds=0.1)
    if result:
        assert result[0]["time"] == 0.25


def test_short_to_segments():

    # Create a short testing result (with time)
    result = [{"chord": "C", "time": 0, "score": 1},
              {"chord": "D", "time": 0.1, "score": 1},
              {"chord": "E", "time": 0.2, "score": 1}]

    segments = sliding_frame.to_segments(result, 0.1)
    assert isinstance(segments, list)
    if segments:
        assert "chord" in segments[0]
        assert "start" in segments[0]
        assert "end" in segments[0]

def test_long_to_segments():
    result = [{"chord": "C", "time": 0, "score": 1},
              {"chord": "C", "time": 0.1, "score": 1},
              {"chord": "C", "time": 0.2, "score": 1},
              {"chord": "C", "time": 0.3, "score": 1},
              {"chord": "D", "time": 0.4, "score": 1},
              {"chord": "D", "time": 0.5, "score": 1},
              {"chord": "D", "time": 0.6, "score": 1},
              {"chord": "D", "time": 0.7, "score": 1}]
    segments = sliding_frame.to_segments(result, 0.1)

    assert len(segments) == 2
    assert segments[0]["chord"] == "C"
    assert segments[1]["chord"] == "D"
    assert segments[0]["end"] == pytest.approx(segments[1]["start"])   # contiguous
    assert segments[0]["end"] == pytest.approx(0.35)


# Test smoothing function when the window is default but input is [C,C,D,C,C]
def test_smooth():
    result = [{"chord": "C", "time": 0, "score": 1},
              {"chord": "C", "time": 0.1, "score": 1},
              {"chord": "D", "time": 0.2, "score": 1},
              {"chord": "C", "time": 0.3, "score": 1},
              {"chord": "C", "time": 0.4, "score": 1}]
    smoothed = sliding_frame.smooth_results(result)
    assert smoothed[2]["chord"] == "C"
    assert smoothed[0]["chord"] == "C"
    assert smoothed[4]["chord"] == "C"

# Test the case where there is a tie with smoothing
def test_smooth_tie():
    result = [{"chord": "C", "time": 0, "score": 1},
              {"chord": "C", "time": 0, "score": 1},
              {"chord": "D", "time": 0.1, "score": 1},
              {"chord": "C", "time": 0.2, "score": 1},
              {"chord": "D", "time": 0.3, "score": 1},
              {"chord": "D", "time": 0.4, "score": 1},
              ]
    smoothed = sliding_frame.smooth_results(result)
    assert smoothed[1]["chord"] == "C"

# Test from a harmonic_note to chords_over_time to smooth_results to to_segments
def test_full_pipeline():
    """
    Test full chord pipeline
    """

    # Generate 3 harmonic notes and stick them to the same signal
    # We begin by crafting the C major chord
    notes_C = sine.harmonic_note(261, 1, damped=True)
    notes_E = sine.harmonic_note(330, 1, damped=True)
    notes_G = sine.harmonic_note(392, 1, damped=True)

    # Next the G chord
    notes_G = sine.harmonic_note(392, 1, damped=True)
    notes_B = sine.harmonic_note(494, 1, damped=True)
    notes_D = sine.harmonic_note(294, 1, damped=True)

    # Next the A minor chord
    notes_A = sine.harmonic_note(440, 1, damped=True)
    # both C and E are added

    # To create this signal, add the C major chord together, wait a bit, then add G and Am
    notes_C_chord = (notes_C + notes_E + notes_G) / 3
    notes_G_chord = (notes_G + notes_B + notes_D) / 3
    notes_Am_chord = (notes_A + notes_C + notes_E) /3

    # Chords cannot be combined into one at once, there needs to be gaps between them
    signal = np.concatenate([notes_C_chord, [0]*1000, notes_G_chord, [0]*1000, notes_Am_chord])

    result = sliding_frame.chords_over_time(signal, 44100, frame_seconds=0.5, hop_seconds=0.1)
    smoothed = sliding_frame.smooth_results(result)
    segments = sliding_frame.to_segments(smoothed, 0.1)

    assert isinstance(segments, list)
    assert [s["chord"] for s in segments] == ["C", "G", "Am"]
    assert segments[0]["end"] == pytest.approx(1.0, abs=0.15)
    assert segments[1]["end"] == pytest.approx(2.0, abs=0.15)




