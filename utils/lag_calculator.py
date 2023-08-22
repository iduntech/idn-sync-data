import numpy as np
import pandas as pd
import mne
import glob
import os
from scipy.signal import resample
from matplotlib import pyplot as plt
from idun_sdk import do_bandpass, prepare_fft
from data_labeler import calculate_bad_epochs
from scipy.signal import find_peaks
import copy

def calculate_lag(signal_1, signal_2):
    '''
    This function calculates the lag between two signals.
    :param signal_1: first signal
    :param signal_2: second signal
    '''
    # check if there is nan in the signals
    if np.isnan(signal_1).any() or np.isnan(signal_2).any():
        # create a nan array with the same length as the signals
        nan_array = np.empty(len(signal_1))
        nan_array[:] = np.nan
        return nan_array, np.nan
    correlation = np.correlate(signal_1, signal_2, mode="full")
    max_correlation = np.max(correlation)
    index_of_max_corr = np.argmax(correlation)
    num_samples = len(signal_1)
    lag = index_of_max_corr - (num_samples - 1)
    return correlation, max_correlation, lag

def calculate_epochs_lag(base_epochs, compare_epochs):
    correlation_arr = []
    lag_arr = []
    max_corr_arr = []
    for idx, epoch in enumerate(base_epochs):
        corr, max_corr, lag = calculate_lag(epoch, compare_epochs[idx])
        max_corr_arr.append(max_corr)
        correlation_arr.append(corr)
        lag_arr.append(lag)
    return correlation_arr, max_corr_arr, lag_arr

def epoch_data(data, samples_per_epoch):
    """Split the data into epochs."""
    num_epochs = len(data) // samples_per_epoch
    # remove the last epoch if it is not complete
    data = data[:num_epochs * samples_per_epoch]
    return np.array_split(data, num_epochs)

def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 -  IQR
    upper_bound = Q3 + IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

def replace_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1  - 0.5*IQR
    upper_bound = Q3 + 0.5*IQR
    # Use np.where to replace outliers with np.nan
    cleaned_data = np.where((data < lower_bound) | (data > upper_bound), np.nan, data)
    return cleaned_data