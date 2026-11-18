"""
Listen for guitar chords in a terminal until C or Ctrl+C is pressed.

Run from the repository root with: python -m audio.live_tools
"""

import argparse
from collections import deque
from contextlib import contextmanager
import os
import select
import sys

import numpy as np
import sounddevice as sd

from audio.recording import list_input_devices
from dsp.mono_tools import to_mono
from pipeline.sliding_frame import analyze_frame, smooth_live_result


@contextmanager
def _terminal_keyboard():
    """Read a stop key without Enter and always restore the terminal settings."""
    
    # Ensure we're in a terminal
    if not sys.stdin.isatty():
        raise RuntimeError('Run live listening in a terminal, not a notebook cell or redirected input.')

    # These modules are available on macOS/Linux. No keyboard-monitoring package
    # or system-wide keyboard permission is needed for the focused terminal.
    import termios
    import tty

    # Get terminal descriptor and save original settings
    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)

    # Function to check if stop key pressed (C)
    def stop_requested():
        ready, _, _ = select.select([descriptor], [], [], 0)
        if not ready:
            return False
        keys = os.read(descriptor, 1024)
        return not keys or b'c' in keys.lower()

    # Set terminal to cbreak mode and yield control
    try:
        tty.setcbreak(descriptor)
        yield stop_requested
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)

# Show status line in terminal
def _show_status(message):
    """Replace the current terminal line instead of printing a growing log."""
    print('\r' + message.ljust(64), end='', flush=True)

# Listen live from mic and show chords
def listen_live(*, device=None, sample_rate=None, min_score=0.8, min_rms=0.005,
                smoothing_window=3):
    """
    Display chord guesses from overlapping half-second microphone frames.

    Read 0.1 seconds at a time, keeping only the latest 0.5 seconds. Reuse the
    existing analyze_frame pipeline, which disables onset detection per frame.

    min_rms is an absolute input-level gate: quieter frames display Silence.
    min_score is the cosine acceptance cutoff. Press C/c without Enter, or
    Ctrl+C, to stop. Run this in a focused macOS/Linux terminal, not Jupyter.

    RMS is required to exceed min_rms to show chords.
    smoothing_window is the number of recent predictions to vote over (odd,
    positive; 1 disables smoothing). Silence and input gaps clear the history.
    """

    # -- INPUT VALIDATION -- #

    # Validate the inputs
    if not np.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError('min_score must be between 0 and 1')
    if not np.isfinite(min_rms) or min_rms < 0:
        raise ValueError('min_rms must be nonnegative and finite')
    if (not isinstance(smoothing_window, (int, np.integer))
            or smoothing_window < 1 or smoothing_window % 2 == 0):
        raise ValueError('smoothing_window must be a positive odd integer')
    
    # sample rate default query
    if sample_rate is None:
        # This is where we get default sample rate if not provided
        sample_rate = sd.query_devices(device, kind='input')['default_samplerate']
    if not np.isfinite(sample_rate) or sample_rate <= 0 or int(sample_rate) != sample_rate:
        raise ValueError('sample_rate must be a positive whole number of Hz')

    # -- SLIDING WINDOW BUFFER -- #

    # Prepare buffer for audio
    sample_rate = int(sample_rate)

    # Frame and hop sizes
    frame_samples = round(0.5 * sample_rate)
    hop_samples = round(0.1 * sample_rate)

    # Check frame/hop sizes
    if frame_samples < 3 or hop_samples < 1:
        raise ValueError('sample_rate is too small for live analysis')

    # Mono capture still arrives as (samples, 1); to_mono removes that axis.
    sd.check_input_settings(device=device, samplerate=sample_rate, channels=1, dtype='float32')
    buffer = np.zeros(frame_samples)
    filled = 0
    # Separate from the audio buffer: this holds only recent chord predictions.
    recent_results = deque(maxlen=int(smoothing_window))

    # -- MAIN LOOP (MICROPHONE) -- #

    # This is the main loop for live audio analysis, the microphone loop
    try:

        # Start live listening loop, read blocks from mic
        # Loop until stop requested, _terminal_keyboard handles this
        with _terminal_keyboard() as stop_requested:

            # Inform user
            print(f'Listening at {sample_rate} Hz. Press C or Ctrl+C to stop.')

            # Open input stream, this is the microphone part.
            with sd.InputStream(
                device=device, samplerate=sample_rate, channels=1,
                dtype='float32', blocksize=hop_samples,
            ) as stream:

                # Read blocks from mic
                while not stop_requested():
                    # Blocking reads keep this first version simple. Only wait
                    # for a tenth of a second, then check the keyboard again.
                    block, overflowed = stream.read(hop_samples)
                    
                    # Check if user wants to stop
                    if stop_requested():
                        break

                    if overflowed:
                        # Missing samples break continuity; refill from fresh
                        # audio instead of joining samples across the gap.
                        buffer.fill(0)
                        filled = 0
                        recent_results.clear()
                        _show_status('Input overflow: refilling the analysis window...')
                        continue

                    # Add new block to buffer
                    block = to_mono(block)
                    count = len(block)

                    if count >= frame_samples:
                        buffer[:] = block[-frame_samples:]

                    # If block is large enough, just take last frame
                    elif count:
                        buffer[:-count] = buffer[count:]
                        buffer[-count:] = block

                    # Filled amount of buffer
                    filled = min(frame_samples, filled + count)

                    # Wait for buffer to fill
                    if filled < frame_samples:
                        _show_status('Collecting the first half-second of audio...')
                        continue

                    # Compute RMS, RMS is needed to check for silence
                    rms = float(np.sqrt(np.mean(buffer ** 2)))

                    # Check for silence
                    if rms <= min_rms:
                        recent_results.clear()
                        label, score = 'Silence', 0.0
                    else:
                        # An actual result was obtained, so analyze it, this is where the pipeline runs
                        result = analyze_frame(buffer, sample_rate, min_score=min_score)
                        recent_results.append({'chord': result['chord'], 'score': result['score']})
                        
                        # Smooth the recent results
                        smoothed = smooth_live_result(recent_results)
                        label = smoothed['chord'] or '?'
                        score = smoothed['score']

                    # Show status for this frame
                    _show_status(f'Chord: {label:<8} Score: {score:.3f}  RMS: {rms:.4f}')

    except KeyboardInterrupt: # Stop on Ctrl+C
        pass
    finally:
        # Stream and keyboard contexts have already cleaned up on every exit.
        print()
    print('Stopped.')


def main():

    # Parse command line arguments, these are not required. More for debugging
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-devices', action='store_true', help='List inputs without recording.')
    parser.add_argument('--device', help='Input device ID or name; default is the system microphone.')
    parser.add_argument('--sample-rate', type=int, help='Override the input device sample rate.')
    parser.add_argument('--min-score', type=float, default=0.8, help='Minimum accepted cosine score.')
    parser.add_argument('--min-rms', type=float, default=0.005, help='Input level below which to show Silence.')
    parser.add_argument('--smoothing-window', type=int, default=3,
                        help='Recent guesses in the majority vote (positive odd; 1 disables smoothing).')
    args = parser.parse_args()
    
    # Start main logic
    try:
        # List devices if requested
        if args.list_devices:

            # List all input devices
            for device in list_input_devices():
                print(device['index'], device['name'], device['default_samplerate'])
            return

        # Determine device to use
        device = int(args.device) if args.device and args.device.isdecimal() else args.device

        # Show chosen device and start listening
        listen_live(device=device, sample_rate=args.sample_rate, min_score=args.min_score,
                    min_rms=args.min_rms, smoothing_window=args.smoothing_window)
    
    except (ValueError, RuntimeError, sd.PortAudioError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
