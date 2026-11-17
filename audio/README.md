The Mac's microphone or audio interface already performs analog-to-digital
conversion (ADC). `sounddevice` delivers its digital samples to Python. This
folder handles recording; DSP stays in `dsp/` and `pipeline/`.

Install the dependency in the Python environment used by your notebook:

```python
%pip install sounddevice
```

On macOS, pip's sounddevice package includes PortAudio. Microphone access is
controlled in System Settings → Privacy & Security → Microphone. Allow the app
that launches your Python/Jupyter process when macOS asks for microphone access.

Sources: [sounddevice installation](https://python-sounddevice.readthedocs.io/en/latest/installation.html),
[Apple microphone access](https://support.apple.com/guide/mac-help/mchla1b1e1fe/mac).

First, list available inputs (this does not record):

```python
from audio.recording import list_input_devices

for device in list_input_devices():
    print(device['index'], device['name'], device['default_samplerate'])
```

Then, when ready to play a chord, run a separate cell:

```python
from audio.recording import record_audio

print('Recording for one second: play and hold your chord now.')
samples, sample_rate = record_audio(duration=1.0)
print('Finished:', samples.shape, sample_rate, 'samples per second')
```

`device=None` uses the default microphone. To select a specific input, pass its
ID or name, for example `record_audio(duration=1.0, device='MacBook Pro Microphone')`
if that is the name in your device list. The function defaults to one channel;
`channels=2` requests stereo when the selected device supports it.

The returned samples have shape `(sample_count, channels)`, including `(N, 1)`
for mono. With 48,000 Hz and one second, that is `(48000, 1)`. Recording waits
until all samples arrive. Importing the module alone does not record anything.

Attach the recording to the same analysis stages as the synthesized examples:

```python
from pipeline.chord_guess import prepare_frame, signal_to_peaks, guess_chord_chroma
from dsp.chroma_tools import build_chroma
from dsp.template_tools import match_chord

frame, weights = prepare_frame(samples, frame_size=sample_rate)
peaks = signal_to_peaks(frame, sample_rate, window_weights=weights)
name, score = match_chord(build_chroma(peaks))
print('Best template:', name, 'score:', score)
print('Accepted chord:', guess_chord_chroma(peaks))
```

Use the returned sample rate throughout; assuming 44,100 Hz for audio captured
at 48,000 Hz would put the FFT frequencies at the wrong pitches. Start with
one held chord per clip: this example analyzes its first second rather than
tracking chord changes. Real microphone noise and guitar harmonics can still
produce incorrect matches; a cosine score is a measure of fit, not certainty.

Keep the recording step separate from automatic notebook runs until you intend
to capture live audio. `record_audio` keeps samples in memory; the file helpers
below save a clip only when you call them. Neither module uploads audio.

Saving is optional for recording, but useful for testing the detector. Two
performances of the same chord differ in timing, level, and background noise.
If you change a template and record again, you cannot tell which difference
caused the new result. Save one raw clip and reuse it to compare code changes
on exactly the same input. The WAV also stores its sample rate, and survives
a notebook kernel restart.

Save the original microphone samples before mono conversion or Hann windowing:

```python
from audio.file_tools import save_audio, load_audio

clip_path = save_audio('recordings/c-major.wav', samples, sample_rate)
print('Saved:', clip_path)
```

This uses uncompressed float32 WAV, preserving the microphone's float32 sample
values, rate, and channels. It does not normalize the recording. Saving to the
same name replaces the previous clip. `recordings/` is ignored by Git.

Sources: [SciPy WAV writing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html),
[SciPy WAV reading](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html).

Later, load and analyze that file without using the microphone:

```python
from audio.file_tools import load_audio
from pipeline.chord_guess import prepare_frame, signal_to_peaks, guess_chord, guess_chord_chroma
from dsp.chroma_tools import build_chroma
from dsp.template_tools import match_chord

saved_samples, saved_rate = load_audio('recordings/c-major.wav')
frame, weights = prepare_frame(saved_samples, frame_size=saved_rate)
peaks = signal_to_peaks(frame, saved_rate, window_weights=weights)
name, score = match_chord(build_chroma(peaks))

print('Exact lookup:', guess_chord(peaks))
print('Best template:', name, 'score:', score)
print('Accepted chord:', guess_chord_chroma(peaks))
```

The load-and-analyze cell can be rerun whenever you edit the detector. Use
one held chord per clip while testing; this still analyzes only the first
second of a longer recording. A wrong chord label is a detector result, not
by itself evidence that microphone capture or WAV loading failed.
