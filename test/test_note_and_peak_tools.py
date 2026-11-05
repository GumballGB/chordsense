import numpy as np

from dsp.note_tools import frequency_to_note
from dsp.peak_tools import find_spectrum_peaks


def test_frequency_to_note_a4_exact():
    info = frequency_to_note(440.0)

    assert info["note"] == "A4"
    assert info["midi"] == 69
    assert abs(info["delta_hz"]) < 1e-12
    assert abs(info["cents"]) < 1e-12


def test_frequency_to_note_a4_sharp():
    info = frequency_to_note(444.0)

    assert info["note"] == "A4"
    assert np.isclose(info["delta_hz"], 4.0, atol=1e-12)
    assert info["cents"] > 0


def test_find_spectrum_peaks_three_tones():
    freqs = np.linspace(0, 500, 501)
    amps = np.zeros_like(freqs)

    amps[131] = 0.30
    amps[165] = 0.33
    amps[196] = 0.31

    peaks = find_spectrum_peaks(
        freqs,
        amps,
        min_height=0.1,
        min_prominence=0.1,
        min_distance_hz=10.0,
        max_peaks=3,
    )

    peak_freqs = sorted(round(p["frequency_hz"]) for p in peaks)
    assert peak_freqs == [131, 165, 196]
