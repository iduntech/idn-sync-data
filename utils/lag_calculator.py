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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from idun_sdk import do_bandpass, prepare_fft, do_highpass
import copy


def calculate_lag(signal_1, signal_2):
    """
    This function calculates the lag between two signals.
    :param signal_1: first signal
    :param signal_2: second signal
    """
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
    data = data[: num_epochs * samples_per_epoch]
    return np.array_split(data, num_epochs)


def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR
    upper_bound = Q3 + IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]


def replace_outliers(data, strictness=0.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - strictness * IQR
    upper_bound = Q3 + strictness * IQR
    # Use np.where to replace outliers with np.nan
    cleaned_data = np.where((data < lower_bound) | (data > upper_bound), np.nan, data)
    return cleaned_data


def prepare_prodigy_data(prodigy_raw_data, config):
    prodigy_data, _ = prodigy_raw_data[:, :]
    prodigy_channel_names = prodigy_raw_data.ch_names

    prodigy_data = pd.DataFrame(prodigy_data.T, columns=prodigy_channel_names)
    prodigy_base_data_resampled = resample_all_prodigy_data(prodigy_data, config)

    prodigy_channel_1_data = np.array(prodigy_base_data_resampled[config.CHANNEL_1])
    prodigy_channel_2_data = np.array(prodigy_base_data_resampled[config.CHANNEL_2])

    # minus right eye from left eye
    prodigy_channel_1_minus_2 = prodigy_channel_1_data - prodigy_channel_2_data
    prodigy_channel_1_minus_2 = (
        prodigy_channel_1_minus_2 * 1000000
    )  # To get the data to same scale as ours, v to uv

    prodigy_filtered_data_rs = do_bandpass(
        prodigy_channel_1_minus_2,
        [config.FILTER_RANGE[0], config.FILTER_RANGE[1]],
        config.BASE_SAMPLE_RATE,
    )
    resampled_times = np.linspace(
        0,
        len(prodigy_filtered_data_rs) / config.BASE_SAMPLE_RATE,
        len(prodigy_filtered_data_rs),
    )

    # create a pandas dataframe with prodigy_channel_names as column names and prodigy_data
    prodigy_data = pd.DataFrame(prodigy_data.T, columns=prodigy_channel_names)
    return prodigy_base_data_resampled, prodigy_filtered_data_rs, resampled_times


def resample_all_prodigy_data(prodigy_base_data_df, config):
    # create a copy of the prodigy_base_data_df
    prodigy_base_data_resampled = []
    # extract one column for lenght calculatoion
    len_channel = len(prodigy_base_data_df[config.CHANNEL_1])
    num_samples_250 = int(
        config.BASE_SAMPLE_RATE / config.PRODIGY_SAMPLE_RATE * len_channel
    )
    for column in prodigy_base_data_df.columns:
        resampled_prodigy_data = resample(prodigy_base_data_df[column], num_samples_250)
        prodigy_base_data_resampled.append(resampled_prodigy_data)
    # convert the list to a numpy array
    prodigy_base_data_resampled = np.array(prodigy_base_data_resampled)
    # transpose the array
    prodigy_base_data_resampled = prodigy_base_data_resampled.T
    # create a pandas dataframe with prodigy_channel_names as column names and prodigy_data
    prodigy_base_data_resampled = pd.DataFrame(
        prodigy_base_data_resampled, columns=prodigy_base_data_df.columns
    )
    return prodigy_base_data_resampled


def prepare_idun_data(idun_raw_data, config):
    idun_data = idun_raw_data[:, 1]
    idun_time_stamps = idun_raw_data[:, 0]
    idun_time_stamps = idun_time_stamps - idun_time_stamps[0]

    # bandpass filter
    idun_filtered_data = do_bandpass(
        idun_data,
        [config.FILTER_RANGE[0], config.FILTER_RANGE[1]],
        config.IDUN_SAMPLE_RATE,
    )
    idun_highpassed_data = do_highpass(
        idun_data, config.HIGHPASS_FREQ, config.IDUN_SAMPLE_RATE
    )
    return idun_highpassed_data, idun_filtered_data, idun_time_stamps

def polynomial_regression_on_lag(cleaned_fine_lag_arr,polynomial_degree):
    original_raw_lag = copy.deepcopy(cleaned_fine_lag_arr)
    x_axis_lag = np.arange(len(cleaned_fine_lag_arr)).reshape(-1, 1)
    x_axis_lag_copy = x_axis_lag.copy()
    
    # find where y is not nan
    not_nan_idx = np.where(~np.isnan(original_raw_lag))[0]
    original_raw_lag = original_raw_lag[not_nan_idx]
    x_axis_lag = x_axis_lag[not_nan_idx]
    
    # Transform the features to 2nd degree polynomial features
    poly = PolynomialFeatures(degree=polynomial_degree)
    X_poly = poly.fit_transform(x_axis_lag)
    
    # Create a LinearRegression model and fit it to the polynomial features
    reg = LinearRegression().fit(X_poly, original_raw_lag)
    
    # Predict values
    X_new_poly = poly.transform(x_axis_lag_copy)
    linear_regression_lag = reg.predict(X_new_poly)
    
    return linear_regression_lag





