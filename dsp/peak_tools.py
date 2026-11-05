import numpy as np
from scipy.signal import find_peaks


def find_spectrum_peaks(
    frequencies_hz,
    amplitudes,
    min_height=0.0,
    min_prominence=0.0,
    min_distance_hz=0.0,
    max_peaks=None,
):
    """
    Detect peaks in a 1D amplitude spectrum.

    Parameters:
    frequencies_hz (array-like): Frequency axis in Hz. You can think of these as
    the frequency bins corresponding to the amplitude values. The length of this array
    should match the length of the amplitudes array.

    amplitudes (array-like): Amplitude values for each frequency bin. You can think of these as
    the magnitude of the signal at each frequency. The length of this array should match
    the length of the frequencies_hz array.

    min_height (float): Minimum amplitude threshold.

    min_prominence (float): Minimum prominence threshold. This means that a peak must stand out 
    from its surroundings by at least this amount.

    min_distance_hz (float): Minimum spacing between neighboring peaks in Hz. It ensures
    that nearby peaks are not counted as separate peaks. The function will convert this value
    to the corresponding number of bins based on the frequency resolution of the input data.

    max_peaks (int | None): Limit returned peaks to top-N by amplitude.

    Returns:
    list[dict]: Each item includes index, frequency_hz, amplitude, prominence.
    """

    # Convert inputs to numpy arrays and validate dimensions
    freqs = np.asarray(frequencies_hz, dtype=float)
    amps = np.asarray(amplitudes, dtype=float)

    # Validate that both inputs are 1D and of the same length
    if freqs.ndim != 1 or amps.ndim != 1:
        raise ValueError("frequencies_hz and amplitudes must be 1D")
    if freqs.size != amps.size:
        raise ValueError("frequencies_hz and amplitudes must have the same length")
    if freqs.size == 0:
        return []

    if freqs.size > 1:
        df = float(np.median(np.diff(freqs)))
        distance_bins = max(1, int(round(float(min_distance_hz) / df)))
    else:
        distance_bins = 1

    # The Scipy find_peaks function is used to identify the indices 
    # of peaks in the amplitude spectrum.
    peak_indices, properties = find_peaks(
        amps,
        height=min_height,
        prominence=min_prominence,
        distance=distance_bins,
    )

    # If no peaks are found, return an empty list.
    if peak_indices.size == 0:
        return []

    # Sort the peaks by amplitude in descending order and limit to max_peaks if specified.
    order = np.argsort(amps[peak_indices])[::-1]
    if max_peaks is not None:
        order = order[: max(0, int(max_peaks))]

    result = []
    prominences = properties.get("prominences", np.zeros_like(peak_indices, dtype=float))

    # For each peak, we create a dictionary containing its index, frequency, amplitude, and prominence.
    for oi in order:
        idx = int(peak_indices[oi])
        result.append(
            {
                "index": idx,
                "frequency_hz": float(freqs[idx]),
                "amplitude": float(amps[idx]),
                "prominence": float(prominences[oi]),
            }
        )

    return result
