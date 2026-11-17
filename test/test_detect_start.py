# Test for detect start function

from dsp.window_tools import detect_start
import numpy as np

def test_detect_start_with_no_skip_offset():

    signal = np.zeros(44100)

    assert detect_start(signal, 44100, skip_seconds=0) == 0

    signal[100] = 1
    assert detect_start(signal, 44100, skip_seconds=0) == 100

    signal[200] = 1
    assert detect_start(signal, 44100, skip_seconds=0) == 100

    signal[50] = 1
    assert detect_start(signal, 44100, skip_seconds=0) == 50

def test_detect_start_with_skip():

    signal = np.zeros(44100)
    signal[100] = 1

    assert detect_start(signal, 44100, skip_seconds=0.1) == 4510

def test_detect_start_empty():

    signal = np.zeros(44100)

    assert detect_start(signal, 44100, skip_seconds=0) == 0

def test_detect_start_empty_with_skip():
    signal = np.zeros(44100)
    assert detect_start(signal, 44100, skip_seconds=0.1) == 4410