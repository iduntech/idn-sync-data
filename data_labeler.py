import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
import numpy as np
import mne
from mne.time_frequency import psd_array_multitaper

mne.set_log_level("WARNING")


def calculate_freq_power(eeg_data: np.ndarray, sfreq: int, freq_band: tuple) -> np.ndarray:
    """
    Calculate the average power in a specified frequency band

    :param eeg_data: EEG data in the form of a 1D numpy array
    :param sfreq: Sampling frequency of the EEG data
    :param freq_band: Frequency band to calculate the power in
    :return: Average power in the specified frequency band
    
    """

    # Calculate PSD for the whole dataset
    psd, freqs = psd_array_multitaper(eeg_data, sfreq, verbose=False)
    freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    # Calculate the average power in the specified frequency band
    freq_power = psd[freq_mask].mean() / 100
    return freq_power


def calculate_bad_epochs(
    eeg_data: np.ndarray,
    sfreq: int = 250,
    epoch_length: int = 125,
    freq_band: tuple = [0.5, 5],
    outlier_threshold=3,
) -> np.ndarray:
    """
    Calculate bad epochs based on the frequency power of the epochs
    :param eeg_data: EEG data in the form of a 1D numpy array
    :param sfreq: Sampling frequency of the EEG data
    :param epoch_length: Length of each epoch in seconds
    :param freq_band: Frequency band to calculate the power in
    :param outlier_threshold: Threshold for identifying outliers.
                              IMPORTANT: 
                                For resting state data make the threshold 3
                                For sleep data make the threshold 15
    :return: Array of bad epochs
    """

    # Split the continuous data into epochs
    n_epochs = len(eeg_data) // epoch_length
    epochs = eeg_data[: n_epochs * epoch_length].reshape((n_epochs, epoch_length))

    # Placeholder for freq_power of each epoch
    epoch_freq_powers = np.zeros(n_epochs)

    # Calculate freq_power for each epoch
    for i, epoch in enumerate(epochs):
        epoch_freq_powers[i] = calculate_freq_power(epoch, sfreq, freq_band)

    # Identify 'bad' epochs, those where power is too high or too low
    # Calculate IQR
    Q1 = np.percentile(epoch_freq_powers, 25)
    Q3 = np.percentile(epoch_freq_powers, 75)
    IQR = Q3 - Q1

    # Define outliers as being more than 1.5*IQR below Q1 or above Q3
    # lower_bound = Q1 - upper_threshold * IQR
    upper_bound = Q3 + outlier_threshold * IQR
    bad_epochs = np.where((epoch_freq_powers > upper_bound))[0]

    # Create an array with good as 0 and bad as 1 that is the same length as the number of epochs
    epoch_labels = np.zeros(n_epochs)

    # Set the bad epochs to 1
    epoch_labels[bad_epochs] = 1
    labels = np.repeat(epoch_labels, epoch_length)

    if len(eeg_data) > len(labels):
        padding = np.full(len(eeg_data) - len(labels), -1)
        labels = np.concatenate((labels, padding))

    return labels, epoch_freq_powers
