from collections import deque
from contextlib import contextmanager
import io
import os
import select
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from audio import live_tools


@pytest.fixture
def live_input(monkeypatch):
    """Simulate streams and stop keys without opening the microphone."""
    state = SimpleNamespace(
        blocks=deque(), stop=False, options=None, stream_closed=False,
        keyboard_restored=False, frames=[],
    )

    class Stream:
        def __init__(self, **kwargs):
            state.options = kwargs

        def __enter__(self):
            return self

        def read(self, count):
            if not state.blocks:
                state.stop = True
                return np.zeros((count, 1)), False
            item = state.blocks.popleft()
            if isinstance(item, BaseException):
                raise item
            return item

        def __exit__(self, *args):
            state.stream_closed = True

    @contextmanager
    def keyboard():
        try:
            yield lambda: state.stop
        finally:
            state.keyboard_restored = True

    fake_sd = SimpleNamespace(
        query_devices=lambda *args, **kwargs: {'default_samplerate': 1000.0},
        check_input_settings=lambda **kwargs: None,
        InputStream=Stream,
    )
    monkeypatch.setattr(live_tools, 'sd', fake_sd)
    monkeypatch.setattr(live_tools, '_terminal_keyboard', keyboard)
    return state


def add_blocks(state, values, length=100):
    state.blocks.extend((np.full((length, 1), value), False) for value in values)


def test_live_frames_overlap_and_use_device_sample_rate(live_input, monkeypatch, capsys):
    add_blocks(live_input, range(1, 8))

    def analyze(frame, sample_rate, min_score):
        live_input.frames.append(frame.copy())
        assert sample_rate == 1000
        assert min_score == 0.85
        return {'chord': 'C', 'score': 0.95}

    monkeypatch.setattr(live_tools, 'analyze_frame', analyze)
    live_tools.listen_live(device=2, min_score=0.85)

    assert len(live_input.frames) == 3
    for index, frame in enumerate(live_input.frames):
        np.testing.assert_array_equal(frame, np.repeat(np.arange(1 + index, 6 + index), 100))
    assert live_input.options == dict(device=2, samplerate=1000, channels=1, dtype='float32', blocksize=100)
    assert live_input.stream_closed and live_input.keyboard_restored
    assert 'Chord: C' in capsys.readouterr().out


def test_quiet_input_does_not_produce_a_chord(live_input, monkeypatch, capsys):
    add_blocks(live_input, [0.001] * 5)

    def unexpected_analysis(*args, **kwargs):
        pytest.fail('Quiet frames should not be sent to the matcher')

    monkeypatch.setattr(live_tools, 'analyze_frame', unexpected_analysis)
    live_tools.listen_live()
    assert 'Chord: Silence' in capsys.readouterr().out


def test_input_overflow_refills_before_analyzing(live_input, monkeypatch, capsys):
    add_blocks(live_input, [1] * 3)
    live_input.blocks.append((np.ones((100, 1)), True))
    add_blocks(live_input, [2] * 5)

    def analyze(frame, *args, **kwargs):
        live_input.frames.append(frame.copy())
        return {'chord': None, 'score': 0.5}

    monkeypatch.setattr(live_tools, 'analyze_frame', analyze)
    live_tools.listen_live()
    assert len(live_input.frames) == 1
    np.testing.assert_array_equal(live_input.frames[0], np.full(500, 2))
    assert 'Input overflow' in capsys.readouterr().out


def test_keyboard_interrupt_closes_the_stream_and_restores_keyboard(live_input, capsys):
    live_input.blocks.append(KeyboardInterrupt())
    live_tools.listen_live()
    assert live_input.stream_closed and live_input.keyboard_restored
    assert 'Stopped.' in capsys.readouterr().out


def test_analysis_error_still_cleans_up(live_input, monkeypatch):
    add_blocks(live_input, [1] * 5)

    def broken_analysis(*args, **kwargs):
        raise RuntimeError('Analysis failed')

    monkeypatch.setattr(live_tools, 'analyze_frame', broken_analysis)
    with pytest.raises(RuntimeError, match='Analysis failed'):
        live_tools.listen_live()
    assert live_input.stream_closed and live_input.keyboard_restored


def test_real_pipeline_identifies_synthesized_stream(live_input, capsys):
    sample_rate = 8000
    times = np.arange(4000) / sample_rate
    chord = sum(np.sin(2 * np.pi * frequency * times) for frequency in (261.625565, 329.627557, 391.995436)) / 3
    for start in range(0, 4000, 800):
        live_input.blocks.append((chord[start:start + 800, None], False))

    live_tools.listen_live(sample_rate=sample_rate)
    assert 'Chord: C ' in capsys.readouterr().out


@pytest.mark.parametrize('options', [
    {'min_score': -0.1}, {'min_score': 1.1}, {'min_rms': -1},
    {'min_rms': np.nan}, {'sample_rate': 0}, {'sample_rate': 1},
    {'smoothing_window': 0}, {'smoothing_window': 2}, {'smoothing_window': 1.5},
])
def test_invalid_live_settings(live_input, options):
    with pytest.raises(ValueError):
        live_tools.listen_live(**options)
    assert live_input.options is None


def test_notebook_input_gets_a_clear_terminal_error(monkeypatch):
    monkeypatch.setattr(sys, 'stdin', io.StringIO())
    with pytest.raises(RuntimeError, match='terminal'):
        with live_tools._terminal_keyboard():
            pytest.fail('Non-terminal input should be rejected')


def test_trailing_vote_uses_the_latest_score_for_the_winning_chord():
    results = [
        {'chord': 'C', 'score': 0.91},
        {'chord': 'C', 'score': 0.93},
        {'chord': 'G', 'score': 0.99},
    ]
    smoothed = live_tools.smooth_live_result(results)
    assert smoothed == {'chord': 'C', 'score': 0.93}
    assert smoothed is not results[1]
    assert results[-1] == {'chord': 'G', 'score': 0.99}


def test_trailing_vote_ties_favor_the_most_recent_guess():
    results = [
        {'chord': 'C', 'score': 0.91},
        {'chord': 'G', 'score': 0.93},
        {'chord': 'Am', 'score': 0.92},
    ]
    assert live_tools.smooth_live_result(results) == results[-1]


def test_unknown_predictions_can_win_the_vote():
    results = [
        {'chord': None, 'score': 0.7},
        {'chord': 'C', 'score': 0.9},
        {'chord': None, 'score': 0.6},
    ]
    assert live_tools.smooth_live_result(results) == {'chord': None, 'score': 0.6}
    assert live_tools.smooth_live_result([]) == {'chord': None, 'score': 0.0}


def run_predictions(state, monkeypatch, predictions, **options):
    add_blocks(state, [1] * (4 + len(predictions)))
    predictions = iter(predictions)
    messages = []
    monkeypatch.setattr(live_tools, 'analyze_frame', lambda *args, **kwargs: next(predictions))
    monkeypatch.setattr(live_tools, '_show_status', messages.append)
    live_tools.listen_live(**options)
    return [message for message in messages if message.startswith('Chord:')]


def test_live_vote_ignores_one_brief_incorrect_prediction(live_input, monkeypatch):
    predictions = [
        {'chord': 'C', 'score': 0.91},
        {'chord': 'C', 'score': 0.93},
        {'chord': 'G', 'score': 0.99},
        {'chord': 'C', 'score': 0.94},
        {'chord': 'C', 'score': 0.95},
    ]
    messages = run_predictions(live_input, monkeypatch, predictions)
    assert all(message.startswith('Chord: C ') for message in messages)
    assert 'Score: 0.930' in messages[2]


def test_live_vote_eventually_follows_a_real_chord_change(live_input, monkeypatch):
    predictions = [{'chord': chord, 'score': 0.9} for chord in ['C', 'C', 'C', 'G', 'G']]
    messages = run_predictions(live_input, monkeypatch, predictions)
    assert [message.split()[1] for message in messages] == ['C', 'C', 'C', 'C', 'G']


def test_smoothing_window_one_displays_raw_guesses(live_input, monkeypatch):
    predictions = [{'chord': chord, 'score': 0.9} for chord in ['C', 'G', 'C']]
    messages = run_predictions(live_input, monkeypatch, predictions, smoothing_window=1)
    assert [message.split()[1] for message in messages] == ['C', 'G', 'C']


def test_silence_clears_old_votes_before_playing_resumes(live_input, monkeypatch):
    add_blocks(live_input, [1] * 7 + [0] * 5 + [1] * 3)
    predictions = iter([{'chord': 'C', 'score': 0.9}] * 7 + [{'chord': 'G', 'score': 0.9}] * 3)
    messages = []
    monkeypatch.setattr(live_tools, 'analyze_frame', lambda *args, **kwargs: next(predictions))
    monkeypatch.setattr(live_tools, '_show_status', messages.append)

    live_tools.listen_live()

    silent = next(i for i, message in enumerate(messages) if message.startswith('Chord: Silence'))
    assert messages[silent + 1].startswith('Chord: G ')


def test_input_gap_clears_old_votes(live_input, monkeypatch):
    add_blocks(live_input, [1] * 7)
    live_input.blocks.append((np.ones((100, 1)), True))
    add_blocks(live_input, [1] * 5)
    predictions = iter([{'chord': 'C', 'score': 0.9}] * 3 + [{'chord': 'G', 'score': 0.9}])
    messages = []
    monkeypatch.setattr(live_tools, 'analyze_frame', lambda *args, **kwargs: next(predictions))
    monkeypatch.setattr(live_tools, '_show_status', messages.append)

    live_tools.listen_live()

    assert messages[-1].startswith('Chord: G ')


@pytest.mark.parametrize('key', [b'c', b'C'])
def test_stop_key_works_without_enter_and_terminal_settings_are_restored(monkeypatch, key):
    import termios

    master, slave = os.openpty()
    try:
        with os.fdopen(slave, 'r') as terminal:
            monkeypatch.setattr(sys, 'stdin', terminal)
            original = terminal_settings(terminal.fileno())
            with live_tools._terminal_keyboard() as stop_requested:
                assert not stop_requested()
                os.write(master, key)
                assert select.select([terminal.fileno()], [], [], 1)[0]
                assert stop_requested()
            assert terminal_settings(terminal.fileno()) == original
    finally:
        os.close(master)


def test_terminal_settings_are_restored_after_an_exception(monkeypatch):
    import termios

    master, slave = os.openpty()
    try:
        with os.fdopen(slave, 'r') as terminal:
            monkeypatch.setattr(sys, 'stdin', terminal)
            original = terminal_settings(terminal.fileno())
            with pytest.raises(RuntimeError, match='test failure'):
                with live_tools._terminal_keyboard():
                    raise RuntimeError('test failure')
            assert terminal_settings(terminal.fileno()) == original
    finally:
        os.close(master)


def terminal_settings(descriptor):
    import termios

    settings = termios.tcgetattr(descriptor)
    # macOS can set PENDIN while switching back to canonical input. It marks
    # pending input for reprocessing, rather than an unrestored input mode.
    settings[3] &= ~getattr(termios, 'PENDIN', 0)
    return settings
