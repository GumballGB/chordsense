import numpy as np
import pytest

from dsp.fourier_transform import amplitude_spectrum
from dsp.harmonic_tools import dampen_harmonics
from dsp.sine import harmonic_note, sine


def test_higher_harmonics_start_quieter_and_fade_faster():
    # Constant input exposes the amplitude envelopes without sine zero crossings.
    times = np.array([0.0, 0.5, 1.0])
    fundamental = dampen_harmonics(np.ones(3), times, 1)
    third = dampen_harmonics(np.ones(3), times, 3)

    assert fundamental[0] == 1
    assert third[0] == pytest.approx(1 / 3)
    assert np.all(np.diff(fundamental) < 0)
    assert third[-1] / third[0] < fundamental[-1] / fundamental[0]


def test_zero_decay_keeps_relative_harmonic_strengths():
    signal = harmonic_note(110, 1, damped=True, decay_rate=0)
    amplitudes = amplitude_spectrum(np.fft.rfft(signal), n_samples=signal.size)
    for i in range(1, 6):
        assert amplitudes[110 * i] / amplitudes[110] == pytest.approx(1 / i)


@pytest.mark.parametrize('damped', [False, True])
def test_note_has_requested_length_and_stays_in_audio_range(damped):
    signal = harmonic_note(110, 1, damped=damped)
    assert signal.shape == (44100,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) <= 1


def test_damped_note_loses_energy_over_time():
    signal = harmonic_note(110, 1, damped=True)
    assert np.mean(signal[-4410:] ** 2) < np.mean(signal[:4410] ** 2)


def test_harmonics_above_nyquist_are_not_aliased():
    signal = harmonic_note(6000, 1, n_harmonics=5)
    amplitudes = amplitude_spectrum(np.fft.rfft(signal), n_samples=signal.size)
    assert np.flatnonzero(amplitudes > 1e-6).tolist() == [6000, 12000, 18000]


def test_one_undamped_harmonic_matches_a_sine():
    np.testing.assert_allclose(harmonic_note(110, 1, n_harmonics=1), sine(110, 1), atol=1e-12)


def test_empty_duration():
    assert harmonic_note(110, 0, damped=True).size == 0


@pytest.mark.parametrize('parameters', [
    {'frequency': 0},
    {'frequency': 22050},
    {'duration': -1},
    {'n_harmonics': 0},
    {'sample_rate': 0},
    {'damped': True, 'decay_rate': -1},
])
def test_invalid_note_parameters(parameters):
    arguments = {'frequency': 110, 'duration': 1, **parameters}
    with pytest.raises(ValueError):
        harmonic_note(**arguments)
