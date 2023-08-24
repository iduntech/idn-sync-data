import os
import glob
import logging
import base64
import numpy as np
import scipy
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import savgol_filter
from scipy import interpolate


def do_bandpass(dataset: np.ndarray, filter_range: list, sample_rate=250) -> np.ndarray:
    """
    This function band passes the data_set.

    Parameters:
        dataset (numpy array) : dataset to be band passed
        fs (int) : sampling frequency
        filter_range (list) : list of the filter range

    Returns:
        filtered_data (numpy array) : band passed dataset
    """

    denom, nom = signal.iirfilter(
        int(3),
        [filter_range[0], filter_range[1]],
        btype="bandpass",
        ftype="butter",
        fs=float(sample_rate),
        output="ba",
    )
    filtered_data = signal.filtfilt(b=denom, a=nom, x=dataset, padtype=None)  # type: ignore
    return filtered_data


def prepare_fft(
    dataset: np.ndarray, low_freq: float, high_freq: float, sample_rate=250
):
    """
    This function prepares the fft without the smoothing.

    Parameters:
        dataset (np.ndarray) : The EEG dataset
        low_freq (int) : The low frequency to be used in the FFT.
        high_freq (int) : The high frequency to be used in the FFT.
        sample_rate (int) : The sampling rate of the dataset.

    Returns:
        freq_analyze (np.ndarray) : The frequency points of the prepared FFT.
        amplitude_analyze (np.ndarray) : The amplitude points of the prepared FFT.
    """

    fft_amplitude = ((2) * (abs(fft(dataset) / len(dataset)))) ** 2
    time_len = len(fft_amplitude)
    time_end = time_len / sample_rate
    time_array = np.arange(0, time_end, 1 / sample_rate)
    freq_array = np.arange(0, sample_rate, (sample_rate) / len(time_array))
    start_position = np.where(freq_array > low_freq)[0][0]
    end_position = np.where(freq_array > high_freq)[0][0]
    amplitude_analyze = fft_amplitude[start_position:end_position]
    freq_analyze = freq_array[start_position:end_position]
    return freq_analyze, amplitude_analyze


def do_highpass(dataset: np.ndarray, freq: int, sample_rate=250) -> np.ndarray:
    """
    This function highpasses the `data_set`.

    Parameters:
        data_set (numpy array) : dataset to be highpassed
        freq (int) : highpass frequency
        fs (int) : sampling frequency

    Returns:
        filtered_data (numpy array) : highpassed dataset
    """

    denom, nom = create_filter_coefficients(freq, sample_rate, "highpass", "butter")
    filtered_data = signal.filtfilt(b=denom, a=nom, x=dataset, padtype=None)  # type: ignore
    return filtered_data


def create_filter_coefficients(
    frequency: float, sample_rate: int, band_type="highpass", filter_type="butter"
) -> tuple:
    """
    This function creates the filter coefficients.

    Parameters:
        frequency (int) : frequency of the filter
        sample_rate (int) : sampling frequency of the filter
        band_type (string) : type of the filter, either highpass or lowpass
        filter_type (string) : type of the filter, either butter or cheby1
    Returns:
        denom (numpy array) : numerator coefficients of the filter
        nom (numpy array) : denominator coefficients of the filter
    """

    denom, nom = signal.iirfilter(
        int(3),
        frequency,
        btype=band_type,
        ftype=filter_type,
        fs=float(sample_rate),
        output="ba",
    )
    return denom, nom
