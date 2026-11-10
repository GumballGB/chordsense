import numpy as np
import pytest

from dsp.fourier_transform import amplitude_spectrum


@pytest.mark.parametrize('n_samples', [255, 256])
@pytest.mark.parametrize('real_fft', [False, True])
@pytest.mark.parametrize('windowed', [False, True])
def test_known_tone_amplitude(n_samples, real_fft, windowed):
    t = np.arange(n_samples)
    signal = 0.33 * np.sin(2 * np.pi * 31 * t / n_samples)
    weights = np.hanning(n_samples) if windowed else None
    samples = signal if weights is None else signal * weights
    coefficients = np.fft.rfft(samples) if real_fft else np.fft.fft(samples)

    amplitudes = amplitude_spectrum(coefficients, n_samples=n_samples, window=weights)

    assert amplitudes.shape == (n_samples // 2 + 1,)
    assert amplitudes[31] == pytest.approx(0.33, rel=1e-5)


@pytest.mark.parametrize('n_samples', [255, 256])
def test_dc_and_last_bin_are_scaled_for_original_length(n_samples):
    t = np.arange(n_samples)
    # The last bin is Nyquist only for even N, so it is doubled only for odd N.
    signal = 0.2 + 0.4 * np.cos(2 * np.pi * (n_samples // 2) * t / n_samples)

    amplitudes = amplitude_spectrum(np.fft.rfft(signal), n_samples=n_samples)

    assert amplitudes[0] == pytest.approx(0.2)
    assert amplitudes[-1] == pytest.approx(0.4)


def test_legacy_full_fft_call_still_works():
    signal = 0.7 * np.sin(2 * np.pi * 12 * np.arange(256) / 256)
    assert amplitude_spectrum(np.fft.fft(signal))[12] == pytest.approx(0.7)


def test_empty_spectrum():
    assert amplitude_spectrum([]).size == 0


@pytest.mark.parametrize('weights', [np.ones(7), np.zeros(8), np.full(8, np.nan)])
def test_invalid_window_is_rejected(weights):
    with pytest.raises(ValueError, match='window'):
        amplitude_spectrum(np.zeros(5), n_samples=8, window=weights)
