"""
Prepare audio with mono conversion and Hann windowing, then guess its chord.

Run the synthesized C-major example from the repository root with:
    python -m pipeline.chord_guess
"""

import numpy as np

import dsp.chord_tools as ct
import dsp.chroma_tools as chroma_tools
from dsp.fourier_transform import amplitude_spectrum
import dsp.mono_tools as mono
import dsp.peak_tools as peak
import dsp.note_tools as note
import dsp.sine as sine
import dsp.template_tools as template_tools
import dsp.window_tools as window


def prepare_frame(signal, frame_size=44100):
    """
    Prepare the first frame for an FFT, with each processing step explicit.

    frame_size counts samples, not seconds. Use frame_size=sample_rate for
    one second of audio. Shorter signals are used as-is, without padding.
    Return (windowed_samples, window_weights); pass both to signal_to_peaks.
    """
    mono_signal = mono.to_mono(signal)

    frame = window.take_frame_mono(mono_signal, frame_size=frame_size)

    # Return empty if no frame
    if frame.size == 0:
        return window.apply_hann(frame)

    # Remove the constant offset before the Hann curve can spread it into peaks.
    centered_frame = frame - np.mean(frame)
    return window.apply_hann(centered_frame)


def signal_to_peaks(signal, sample_rate=44100, window_weights=None):
    """
    Compute an amplitude spectrum and find strong peaks in the supplied audio.

    sample_rate must match the rate used to generate or record the signal.
    All supplied samples are analyzed.

    Call prepare_frame first to attach mono
    conversion, framing, and Hann windowing. This function only performs the
    spectrum and peak-detection stages.

    It also accepts unwindowed mono audio.
    Pass the window_weights returned by prepare_frame to correct amplitudes
    for Hann attenuation. Leave them as None for unwindowed audio.
    Mean removal happens only in prepare_frame, before windowing.
    Peaks below 10% of the strongest spectral amplitude are ignored. This
    starting threshold works for synthesized tones; guitar harmonics and noise
    will need further handling before peaks reliably represent played notes.
    """

    # Check sample rate and signal validity
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite")
    samples = np.asarray(signal, dtype=float)
    if samples.ndim != 1:
        raise ValueError("signal must be a 1D mono signal")
    if not np.all(np.isfinite(samples)):
        raise ValueError("signal must contain only finite samples")
    if samples.size < 3:
        return []
    # Constant audio contains only DC. Skip it before FFT roundoff can look
    # like tiny peaks after the DC bin is excluded.
    if np.all(samples == samples[0]):
        return []

    # NumPy's real FFT accepts 44,100 samples and omits negative frequencies.
    spectrum = np.fft.rfft(samples)

    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)

    amplitudes = amplitude_spectrum(
        spectrum, n_samples=samples.size, window=window_weights
    )

    amplitudes[0] = 0  # DC is not a musical note.

    # Find the strongest peak
    strongest = amplitudes.max()

    # Skip if no strong peaks
    if strongest == 0:
        return []

    # Find peaks in the spectrum
    return peak.find_spectrum_peaks(
        frequencies,
        amplitudes,
        min_height=0.1 * strongest,
        min_prominence=0.1 * strongest,
    )


def guess_chord(peaks):
    """
    Map peak dictionaries to MIDI notes, then return a chord name or None.
    """

    # Return empty if no peaks
    if not peaks:
        return None

    # Prepare list for MIDI notes
    midi_notes = []

    # Convert to MIDI notes
    for p in peaks:
        midi_notes.append(note.frequency_to_note(p["frequency_hz"])["midi"])

    # Extract chord from MIDI notes
    return ct.chord_extract(midi_notes)


def guess_chord_chroma(peaks, min_score=0.8):
    """
    Return the best cosine-matched chord name, or None below min_score.

    peaks is a list of frequency/amplitude dicts. They are not notes

    min_score is the cutoff for accepting a chord.

    All chroma weights reach the matcher; no pitch-class cutoff is applied.
    min_score is a heuristic final-answer cutoff, not a confidence probability.
    Set it to 0 to accept any non-silent match. To inspect the score, call
    match_chord(build_chroma(peaks)) from dsp.template_tools/dsp.chroma_tools.
    """

    # Validate min_score.
    if not np.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")

    # Sum amplitudes into 12 pitch-class bins, keeping even weak bins.
    chroma = chroma_tools.build_chroma(peaks)

    # Compare the complete vector against every chord template.
    # A cosine similarity will be used to match chords.
    name, score = template_tools.match_chord(chroma)

    if score < min_score:
        return None
    
    return name


if __name__ == "__main__":

    sample_rate = 44100

    # Add C4, E4, and G4 together: a chord needs several simultaneous notes.
    frequencies = (261.625565, 329.627557, 391.995436)

    signal = 0

    # Add each frequency to the signal
    for f in frequencies:
        signal += sine.sine(f, 1, sample_rate)

    # Normalize the signal
    signal = signal / len(frequencies)

    # Prepare frame for analysis
    frame, weights = prepare_frame(signal, frame_size=sample_rate)

    # Find peaks in the frame
    peaks = signal_to_peaks(frame, sample_rate=sample_rate, window_weights=weights)

    notes = [note.frequency_to_note(p["frequency_hz"])["note"] for p in peaks]
    print("Detected notes:", ", ".join(notes))
    chord = guess_chord(peaks)
    print("Guessed chord:", chord if chord is not None else "?")


