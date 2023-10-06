### This script is required because we want to import .edf files using pyedflib, but want to match
### the resulting format to look like it has been imported by MNE (because MNE was used initally to
### import data).
import numpy as np
from scipy.signal import resample


def pyedflib_to_mne(
    pyedflib_data: np.ndarray,
    chan_name: str,
    target_length: int,
    sample_freq: int,
    dimension: str,
):
    """
    This function resamples channels that are not matching the prodigy target sampling frequency (120Hz).

    Parameters:
        pyedflib_data (numpy array) : one dimensional array to be checked and converted
        chan_name (str) : channel name
        target_length (int) : to which length each array needs to be resampled to (if not already at that length)
        sample_freq (int) : this channels original sampling frequency
        dimension (str) : dimension of this channel (either 'uV' or '' (empty))

    Returns:
        resampled (or untouched) one dimensional array
    """
    if sample_freq != 120:
        converted_data = resample(pyedflib_data, target_length)
    else:
        converted_data = pyedflib_data

    return converted_data


def create_timestamp_array(target_length: int, samp_rate: int):
    """
    This function creates an array that increases in 1/120 increments until its length matches the length of the dataset.

    Parameters:
        target_length (int array) : desired length of the array
        sample_freq (int) : target sampling frequency (e.g., 120Hz)

    Returns:
        Timestamp array, each increment is 1/120
    """
    increment = (
        1 / samp_rate
    )  # increase in these increments based on prodigy EEG channel sampling rate
    timestamp_array = np.arange(0, target_length * increment, increment)

    return timestamp_array
