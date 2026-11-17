# Test for sliding frame chord analysis

import pipeline.sliding_frame as sliding_frame

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

