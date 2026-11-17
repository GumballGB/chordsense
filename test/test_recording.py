from types import SimpleNamespace

import numpy as np
import pytest

from audio import recording
from pipeline.chord_guess import prepare_frame, signal_to_peaks, guess_chord_chroma


@pytest.fixture
def backend(monkeypatch):
    inputs = [
        {'index': 0, 'name': 'Speakers', 'max_input_channels': 0, 'default_samplerate': 48000.0},
        {'index': 1, 'name': 'Test microphone', 'max_input_channels': 2, 'default_samplerate': 48000.0},
    ]
    calls = []

    def query_devices(device=None, kind=None):
        return inputs[1] if kind == 'input' else inputs

    def rec(frames, **kwargs):
        calls.append((frames, kwargs))
        return np.zeros((frames, kwargs['channels']), dtype=np.float32)

    fake = SimpleNamespace(
        query_devices=query_devices,
        check_input_settings=lambda **kwargs: None,
        rec=rec,
        get_status=lambda: SimpleNamespace(input_overflow=False),
        calls=calls,
    )
    monkeypatch.setattr(recording, 'sd', fake)
    return fake


def test_device_list_only_contains_inputs(backend):
    assert [d['name'] for d in recording.list_input_devices()] == ['Test microphone']
    assert backend.calls == []


def test_recording_uses_device_rate_and_waits_for_samples(backend):
    samples, rate = recording.record_audio(duration=0.25, channels=2, device=1)
    assert rate == 48000
    assert samples.shape == (12000, 2)
    assert samples.dtype == np.float32
    frames, options = backend.calls[0]
    assert frames == 12000
    assert options == dict(samplerate=48000, channels=2, dtype='float32', device=1, blocking=True)


def test_explicit_sample_rate_is_checked(backend):
    checked = []
    backend.check_input_settings = lambda **kwargs: checked.append(kwargs)
    samples, rate = recording.record_audio(duration=0.5, sample_rate=16000)
    assert rate == 16000
    assert samples.shape == (8000, 1)
    assert checked[0]['samplerate'] == 16000


def test_unsupported_settings_do_not_start_recording(backend):
    def unsupported(**kwargs):
        raise RuntimeError('Unsupported input settings')

    backend.check_input_settings = unsupported
    with pytest.raises(RuntimeError, match='Unsupported'):
        recording.record_audio()
    assert backend.calls == []


def test_input_overflow_is_reported(backend):
    backend.get_status = lambda: SimpleNamespace(input_overflow=True)
    with pytest.raises(RuntimeError, match='overflowed'):
        recording.record_audio()


@pytest.mark.parametrize('options', [
    {'duration': 0}, {'duration': -1}, {'duration': np.nan}, {'duration': 1e-10},
    {'sample_rate': 0}, {'sample_rate': np.inf}, {'sample_rate': 44100.5},
    {'channels': 0}, {'channels': 1.5},
])
def test_invalid_recording_options(backend, options):
    with pytest.raises(ValueError):
        recording.record_audio(**options)
    assert backend.calls == []


def test_recorded_samples_connect_to_chord_pipeline(backend):
    def simulated_recording(frames, **kwargs):
        times = np.arange(frames) / kwargs['samplerate']
        chord = sum(np.sin(2 * np.pi * frequency * times) for frequency in (261.625565, 329.627557, 391.995436)) / 3
        return chord.astype(np.float32)[:, None]

    backend.rec = simulated_recording
    samples, rate = recording.record_audio()
    frame, weights = prepare_frame(samples, sample_rate=rate, frame_size=rate)
    peaks = signal_to_peaks(frame, rate, window_weights=weights)
    assert guess_chord_chroma(peaks) == 'C'
