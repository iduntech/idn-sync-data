import numpy as np
import pandas as pd
import mne
import glob
import os
from scipy.signal import resample
from matplotlib import pyplot as plt
from utils.freq_calculator import do_bandpass, prepare_fft, do_highpass
from scipy.signal import find_peaks
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pyedflib import highlevel
from utils.edf_format_prep import pyedflib_to_mne, create_timestamp_array
import copy
import pyxdf
import config


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
        try:
            corr, max_corr, lag = calculate_lag(epoch, compare_epochs[idx])
            max_corr_arr.append(max_corr)
            correlation_arr.append(corr)
            lag_arr.append(lag)
        except IndexError:
            print("End of shorter data reached")
            break
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


def convert_data_to_array(comparisoneeg_raw_data, file_extention):
    if file_extention == "edf":
        # Convert the list of arrays into a NumPy array of arrays
        comparisoneeg_data = np.array(comparisoneeg_raw_data[0])

        # comparisoneeg_data, _ = comparisoneeg_raw_data[:, :]
        comparisoneeg_data = comparisoneeg_data.T
        # TODO: Make that input to this function is an array and the channels
        # TODO: This is so that this function will work on full scalp as well as prodigy
        # comparisoneeg_channel_names = comparisoneeg_raw_data.ch_names
        comparisoneeg_channel_names = comparisoneeg_raw_data[1]

    elif file_extention == "xdf":
        comparisoneeg_data = comparisoneeg_raw_data[0]["time_series"]
        n_channel = int(comparisoneeg_raw_data[0]["info"]["channel_count"][0])
        comparisoneeg_channel_names = []
        for item in range(n_channel):
            comparisoneeg_channel_names.append(
                comparisoneeg_raw_data[0]["info"]["desc"][0]["channels"][0]["channel"][
                    item
                ]["label"]
            )
        comparisoneeg_channel_names = [
            item for sublist in comparisoneeg_channel_names for item in sublist
        ]

    comparisoneeg_data = pd.DataFrame(
        comparisoneeg_data, columns=comparisoneeg_channel_names
    )
    return comparisoneeg_data, comparisoneeg_channel_names


def prepare_comparison_data(
    comparisoneeg_data, config
):  # prepare_prodigy_data(ra, config):
    comparisoneeg_base_data_resampled = resample_all_comparisoneeg_data(
        comparisoneeg_data, config
    )
    channel_1_key, channel_2_key, scale_factor, sample_rate = get_device_configuration(
        config
    )

    # Extract channel data based on the current device configuration
    comparisoneeg_channel_1_data = np.array(
        comparisoneeg_base_data_resampled[channel_1_key]
    )
    comparisoneeg_channel_2_data = np.array(
        comparisoneeg_base_data_resampled[channel_2_key]
    )

    # Calculate the difference between channels
    comparisoneeg_channel_1_minus_2 = (
        comparisoneeg_channel_1_data - comparisoneeg_channel_2_data
    )

    # Apply the scale factor
    comparisoneeg_channel_1_minus_2 *= scale_factor

    comparisoneeg_filtered_data_rs = do_bandpass(
        comparisoneeg_channel_1_minus_2,
        [config.FILTER_RANGE[0], config.FILTER_RANGE[1]],
        config.BASE_SAMPLE_RATE,
    )

    resampled_times = np.linspace(
        0,
        len(comparisoneeg_filtered_data_rs) / config.BASE_SAMPLE_RATE,
        len(comparisoneeg_filtered_data_rs),
    )

    return (
        comparisoneeg_base_data_resampled,
        comparisoneeg_filtered_data_rs,
        resampled_times,
    )


def get_device_configuration(config):
    if config.DEVICE == "PRODIGY":
        channel_1_key = config.PRODIGY_CHANNEL_1
        channel_2_key = config.PRODIGY_CHANNEL_2
        scale_factor = config.PRODIGY_SCALE_FACTOR
        sample_rate = config.PRODIGY_SAMPLE_RATE

    elif config.DEVICE == "MBT":
        channel_1_key = config.MBT_CHANNEL_1
        channel_2_key = config.MBT_CHANNEL_2
        scale_factor = config.MBT_SCALE_FACTOR
        sample_rate = config.MBT_SAMPLE_RATE

    else:
        raise ValueError(f"Unsupported device: {config.DEVICE}")

    return channel_1_key, channel_2_key, scale_factor, sample_rate


def resample_all_comparisoneeg_data(comparisoneeg_base_data_df, config):
    # create a copy of the prodigy_base_data_df
    comparisoneeg_base_data_resampled = []
    # extract one column for lenght calculatoion
    channel_1_key, _, _, sample_rate = get_device_configuration(config)
    # (sample,scale,key1,key2) = get_device_configuration(config)
    len_channel = len(comparisoneeg_base_data_df[channel_1_key])
    num_samples_250 = int(config.BASE_SAMPLE_RATE / sample_rate * len_channel)

    for column in comparisoneeg_base_data_df.columns:
        resampled_comparisoneeg_data = resample(
            comparisoneeg_base_data_df[column], num_samples_250
        )
        comparisoneeg_base_data_resampled.append(resampled_comparisoneeg_data)
    # convert the list to a numpy array
    comparisoneeg_base_data_resampled = np.array(comparisoneeg_base_data_resampled)
    # transpose the array
    comparisoneeg_base_data_resampled = comparisoneeg_base_data_resampled.T
    # create a pandas dataframe with prodigy_channel_names as column names and prodigy_data
    comparisoneeg_base_data_resampled = pd.DataFrame(
        comparisoneeg_base_data_resampled, columns=comparisoneeg_base_data_df.columns
    )

    return comparisoneeg_base_data_resampled


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


def best_polynomial_regression_on_lag(cleaned_fine_lag_arr, config):
    original_raw_lag = copy.deepcopy(cleaned_fine_lag_arr)
    x_axis_lag = np.arange(len(cleaned_fine_lag_arr)).reshape(-1, 1)
    x_axis_lag_copy = x_axis_lag.copy()

    # find where y is not nan
    not_nan_idx = np.where(~np.isnan(original_raw_lag))[0]
    original_raw_lag = original_raw_lag[not_nan_idx]
    x_axis_lag = x_axis_lag[not_nan_idx]

    # Rest of your code
    polynomial_degree = config.POLYNOMIAL_ORDER
    best_degree = None
    best_mae = float("inf")
    best_linear_regression_lag = []

    for degree in polynomial_degree:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(x_axis_lag)
        # Create a LinearRegression model and fit it to the polynomial features
        reg = LinearRegression().fit(X_poly, original_raw_lag)
        # Predict values
        X_new_poly = poly.transform(x_axis_lag_copy)
        linear_regression_lag = reg.predict(X_new_poly)

        # Calculate the mean absolute error
        mae = mean_absolute_error(x_axis_lag_copy, linear_regression_lag)
        print("Degree:", degree, " , MAE:", mae)

        if mae < best_mae:
            best_mae = mae
            best_degree = degree
            best_linear_regression_lag = linear_regression_lag

    print("Best Polynomial Degree:", best_degree)
    print("Best Mean Absolute Error:", best_mae)

    return best_degree


def polynomial_regression_on_lag(best_degree, cleaned_fine_lag_arr):
    original_raw_lag = copy.deepcopy(cleaned_fine_lag_arr)
    x_axis_lag = np.arange(len(cleaned_fine_lag_arr)).reshape(-1, 1)
    x_axis_lag_copy = x_axis_lag.copy()

    # find where y is not nan
    not_nan_idx = np.where(~np.isnan(original_raw_lag))[0]
    original_raw_lag = original_raw_lag[not_nan_idx]
    x_axis_lag = x_axis_lag[not_nan_idx]

    poly = PolynomialFeatures(degree=best_degree)
    X_poly = poly.fit_transform(x_axis_lag)
    # Create a LinearRegression model and fit it to the polynomial features
    reg = LinearRegression().fit(X_poly, original_raw_lag)
    # Predict values
    X_new_poly = poly.transform(x_axis_lag_copy)
    linear_regression_lag = reg.predict(X_new_poly)

    return linear_regression_lag


def cut_throughout_data(
    prodigy_cut_data, prodigy_base_cut_df, lag_positions, cumulative_lags
):
    prodigy_cut_data_list = prodigy_cut_data.tolist()
    try:
        for i in range(len(lag_positions)):
            start_index = lag_positions[i]
            # Calculate how many elements to replace with np.nan
            if i == 0:
                n_replace = int(cumulative_lags[i])
            else:
                n_replace = int(cumulative_lags[i] - cumulative_lags[i - 1])

            for j in range(n_replace):
                if start_index + j < len(prodigy_cut_data_list):
                    prodigy_cut_data_list[start_index + j] = np.nan
                    prodigy_base_cut_df.iloc[start_index + j] = np.nan
    except IndexError:
        pass
    # Convert back to numpy array and remove np.nan values
    prodigy_cut_data_list = np.array(prodigy_cut_data_list)
    prodigy_cut_data_list = prodigy_cut_data_list[~np.isnan(prodigy_cut_data_list)]
    # Drop rows in base_dataset2 that contain NaN values
    prodigy_base_cut_df = prodigy_base_cut_df.dropna()
    return prodigy_cut_data_list, prodigy_base_cut_df


def cut_throughout_data_dual(
    idun_cut_data,
    idun_base_cut_data,
    comparison_cut_data,
    comparison_base_cut_df,
    lag_positions,
    lag_sizes,
):
    idun_cut_data_list = idun_cut_data.tolist()
    comparison_cut_data_list = comparison_cut_data.tolist()
    for i in range(len(lag_positions)):
        start_index = lag_positions[i]

        if lag_sizes[i] < 0:
            # Negative lag_size: prune IDUN data
            n_replace = abs(lag_sizes[i])
            for j in range(n_replace):
                if start_index + j < len(idun_cut_data_list):
                    idun_cut_data_list[start_index + j] = np.nan
                    idun_base_cut_data[start_index + j] = np.nan
        else:
            # Positive lag_size: prune comparison data
            n_replace = lag_sizes[i]
            for j in range(n_replace):
                if start_index + j < len(comparison_cut_data_list):
                    comparison_cut_data_list[start_index + j] = np.nan
                    comparison_base_cut_df.iloc[start_index + j] = np.nan

    # Convert back to numpy arrays and remove np.nan values
    idun_cut_data_list = np.array(idun_cut_data_list)
    remove_positions = ~np.isnan(idun_cut_data_list)
    idun_cut_data_list = idun_cut_data_list[remove_positions]
    idun_base_cut_data = idun_base_cut_data[remove_positions]

    comparison_cut_data_list = np.array(comparison_cut_data_list)
    comparison_cut_data_list = comparison_cut_data_list[
        ~np.isnan(comparison_cut_data_list)
    ]
    comparison_base_cut_df = comparison_base_cut_df.dropna()

    # cut at the end of the longer datasets
    if len(comparison_cut_data_list) > len(idun_cut_data_list):
        comparison_cut_data_list = comparison_cut_data_list[: len(idun_cut_data_list)]
        comparison_base_cut_df = comparison_base_cut_df[: len(idun_cut_data_list)]
    else:
        idun_cut_data_list = idun_cut_data_list[: len(comparison_cut_data_list)]
        idun_base_cut_data = idun_base_cut_data[: len(comparison_cut_data_list)]

    return (
        idun_cut_data_list,
        idun_base_cut_data,
        comparison_cut_data_list,
        comparison_base_cut_df,
    )


def cut_throughout_data_arr(
    idun_cut_data, idun_base_cut, lag_positions, cumulative_lags
):
    idun_cut_data_list = idun_cut_data.tolist()
    idun_base_cut_list = (
        idun_base_cut.tolist()
    )  # Convert to list for easier manipulation
    try:
        for i in range(len(lag_positions)):
            start_index = lag_positions[i]
            # Calculate how many elements to replace with np.nan
            if i == 0:
                n_replace = int(cumulative_lags[i])
            else:
                n_replace = int(cumulative_lags[i] - cumulative_lags[i - 1])
            for j in range(n_replace):
                if start_index + j < len(idun_cut_data_list):
                    idun_cut_data_list[start_index + j] = np.nan
                    idun_base_cut_list[
                        start_index + j
                    ] = np.nan  # Modify list in the same manner
    except IndexError:
        pass
    # Convert back to numpy array and remove np.nan values
    idun_cut_data_array = np.array(idun_cut_data_list)
    idun_cut_data_array = idun_cut_data_array[~np.isnan(idun_cut_data_array)]
    # Convert base_cut_list back to numpy array and remove np.nan values
    idun_base_cut_array = np.array(idun_base_cut_list)
    idun_base_cut_array = idun_base_cut_array[~np.isnan(idun_base_cut_array)]
    return idun_cut_data_array, idun_base_cut_array


def interpolate_signal(raw_signal):
    # deepcopy the signal
    signal = copy.deepcopy(raw_signal)
    # Convert the signal into a pandas Series
    s = pd.Series(signal)

    # Interpolate the nan values using linear method
    s.interpolate(method="linear", inplace=True)

    # Backfill and forward fill for the start and end nans respectively
    s.fillna(method="bfill", inplace=True)
    s.fillna(method="ffill", inplace=True)

    return s.values


def smooth(data, window_size=3):
    if window_size < 2:  # No smoothing needed
        return data
    s = np.r_[data[window_size - 1 : 0 : -1], data, data[-2 : -window_size - 1 : -1]]
    w = np.ones(window_size, "d")
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[int((window_size / 2) - 1) : -int(window_size / 2)]


def cut_at_end(
    comparisoneeg_adjusted_final_arr,
    comparisoneeg_adjusted_base_final_df,
    idun_cut_data,
    idun_base_cut_data,
):
    # Cut from the end of the longer dataset
    if len(comparisoneeg_adjusted_final_arr) > len(idun_cut_data):
        comparisoneeg_adjusted_final_arr = comparisoneeg_adjusted_final_arr[
            : len(idun_cut_data)
        ]
        comparisoneeg_adjusted_base_final_df = comparisoneeg_adjusted_base_final_df[
            : len(idun_cut_data)
        ].reset_index(drop=True)

    else:
        idun_cut_data = idun_cut_data[: len(comparisoneeg_adjusted_final_arr)]
        idun_base_cut_data = idun_base_cut_data[: len(comparisoneeg_adjusted_final_arr)]

    return (
        comparisoneeg_adjusted_final_arr,
        comparisoneeg_adjusted_base_final_df,
        idun_cut_data,
        idun_base_cut_data,
    )


def adjust_data_by_mean_lag(
    mean_final_lag,
    comparisoneeg_adjusted_final_arr,
    comparisoneeg_adjusted_base_final_df,
    idun_adjusted_final_arr,
    idun_adjusted_base_final_arr,
):
    if int(mean_final_lag) > 0:
        shifted_final_comparisoneeg_arr = comparisoneeg_adjusted_final_arr[
            int(mean_final_lag) :
        ]
        shifted_final_comparisoneeg_base_df = comparisoneeg_adjusted_base_final_df[
            int(mean_final_lag) :
        ].reset_index(drop=True)
        shifted_final_idun_arr = idun_adjusted_final_arr[: -int(mean_final_lag)]
        shifted_final_idun_base_arr = idun_adjusted_base_final_arr[
            : -int(mean_final_lag)
        ]
    elif int(mean_final_lag) < 0:
        shifted_final_comparisoneeg_arr = comparisoneeg_adjusted_final_arr[
            : -(-int(mean_final_lag))
        ]
        shifted_final_comparisoneeg_base_df = comparisoneeg_adjusted_base_final_df[
            : -(-int(mean_final_lag))
        ].reset_index(drop=True)

        shifted_final_idun_arr = idun_adjusted_final_arr[-int(mean_final_lag) :]
        shifted_final_idun_base_arr = idun_adjusted_base_final_arr[
            -int(mean_final_lag) :
        ]
    else:
        shifted_final_comparisoneeg_arr = comparisoneeg_adjusted_final_arr
        shifted_final_comparisoneeg_base_df = comparisoneeg_adjusted_base_final_df
        shifted_final_idun_arr = idun_adjusted_final_arr
        shifted_final_idun_base_arr = idun_adjusted_base_final_arr

    return (
        shifted_final_comparisoneeg_arr,
        shifted_final_comparisoneeg_base_df,
        shifted_final_idun_arr,
        shifted_final_idun_base_arr,
    )


def sync_data_start_same_time(
    comparisoneeg_clipped_data,
    comparisoneeg_base_clipped_df,
    idun_clipped_data,
    idun_base_clipped_data,
    comparisoneeg_time_stamps,
    idun_time_stamps,
    config,
):
    timestamp_diff = comparisoneeg_time_stamps[0] - idun_time_stamps[0]
    if timestamp_diff >= 0:
        # comparisonEEG starts before IDUN
        comparisoneeg_clipped_data = comparisoneeg_clipped_data[
            int(timestamp_diff / config.BASE_SAMPLE_RATE) : -1
        ]
        comparisoneeg_base_clipped_df = comparisoneeg_base_clipped_df[
            int(timestamp_diff / config.BASE_SAMPLE_RATE) : -1
        ].reset_index(drop=True)
    elif timestamp_diff < 0:
        # IDUN starts before comparisonEEG
        idun_clipped_data = idun_clipped_data[
            int(timestamp_diff / config.BASE_SAMPLE_RATE) : -1
        ]
        idun_base_clipped_data = idun_base_clipped_data[
            int(timestamp_diff / config.BASE_SAMPLE_RATE) : -1
        ].reset_index(drop=True)

    if timestamp_diff == 0 and comparisoneeg_time_stamps[0] == 0:  # prodigy
        next_step = True
    else:
        next_step = False  # mbt
    return (
        comparisoneeg_clipped_data,
        comparisoneeg_base_clipped_df,
        idun_clipped_data,
        idun_base_clipped_data,
        next_step,
    )


def check_timestamp(comparisoneeg_time_stamps, idun_time_stamps):
    timestamp_diff = comparisoneeg_time_stamps[0] - idun_time_stamps[0]
    if timestamp_diff == 0 and comparisoneeg_time_stamps[0] == 0:  # ex. prodigy
        next_step = False
    else:
        next_step = True  # ex. mbt

    return next_step


def cut_ends_to_same_length(
    comparisoneeg_clipped_data,
    comparisoneeg_base_clipped_df,
    idun_clipped_data,
    idun_base_clipped_data,
    config,
):
    # Find which one is longer and how much longer, I've already done the resampling
    if len(comparisoneeg_clipped_data) > len(idun_clipped_data):
        diff = int(len(comparisoneeg_clipped_data) - len(idun_clipped_data))
        # I cut the longer signal only at the end
        comparisoneeg_clipped_data = comparisoneeg_clipped_data[
            0 : len(idun_clipped_data)
        ]
        comparisoneeg_base_clipped_df = comparisoneeg_base_clipped_df[
            0 : len(idun_clipped_data)
        ].reset_index(drop=True)
        print(
            f"Comparison data is longer with {diff/config.BASE_SAMPLE_RATE} seconds, cutting from end of Prodigy data"
        )
    else:
        diff = int(len(idun_clipped_data) - len(comparisoneeg_clipped_data))
        # I cut the longer signal only at the end
        idun_clipped_data = idun_clipped_data[0 : len(comparisoneeg_clipped_data)]
        idun_base_clipped_data = idun_base_clipped_data[
            0 : len(comparisoneeg_clipped_data)
        ]
        print(
            f"IDUN data is longer with {diff/config.BASE_SAMPLE_RATE} seconds, cutting from end of IDUN data"
        )
    return (
        comparisoneeg_clipped_data,
        comparisoneeg_base_clipped_df,
        idun_clipped_data,
        idun_base_clipped_data,
    )


def sync_start_and_equalize_data_length(
    comparisoneeg_filtered_data_rs,
    idun_filtered_data,
    idun_base_data,
    comparisoneeg_base_data_df,
    comparisoneeg_time_stamps,
    idun_time_stamps,
    config,
):
    comparisoneeg_clipped_data = copy.deepcopy(comparisoneeg_filtered_data_rs)
    idun_clipped_data = copy.deepcopy(idun_filtered_data)
    idun_base_clipped_data = copy.deepcopy(idun_base_data)
    comparisoneeg_base_clipped_df = copy.deepcopy(comparisoneeg_base_data_df)

    (
        comparisoneeg_clipped_data,
        comparisoneeg_base_clipped_df,
        idun_clipped_data,
        idun_base_clipped_data,
        next_step,
    ) = sync_data_start_same_time(
        comparisoneeg_clipped_data,
        comparisoneeg_base_clipped_df,
        idun_clipped_data,
        idun_base_clipped_data,
        comparisoneeg_time_stamps,
        idun_time_stamps,
        config,
    )

    (
        comparisoneeg_clipped_data,
        comparisoneeg_base_clipped_df,
        idun_clipped_data,
        idun_base_clipped_data,
    ) = cut_ends_to_same_length(
        comparisoneeg_clipped_data,
        comparisoneeg_base_clipped_df,
        idun_clipped_data,
        idun_base_clipped_data,
        config,
    )

    same_times = np.linspace(
        0, len(idun_clipped_data) / config.BASE_SAMPLE_RATE, len(idun_clipped_data)
    )

    if next_step == True:
        print("Search for a better alignment")
    else:
        print("First alignment completed")

    return (
        comparisoneeg_clipped_data,
        idun_clipped_data,
        idun_base_clipped_data,
        comparisoneeg_base_clipped_df,
        same_times,
        next_step,
    )


def identify_discontinuities(data, threshold):
    diffs = np.diff(data)
    discontinuities = np.where(np.abs(diffs) > threshold)[0]
    return discontinuities


def make_discontinuous(data, discontinuous_indices):
    for index in discontinuous_indices:
        data[index + 1] = np.nan
    return data


def clean_data_from_spikes(data, threshold):
    discontinuous_indices = identify_discontinuities(data, threshold)
    data = make_discontinuous(data, discontinuous_indices)
    data = np.array(data)
    return data


def manual_sync(
    comparison_clipped_data,
    idun_clipped_data,
    idun_base_clipped_data,
    comparison_base_clipped_df,
    config,
    MANUAL_SHIFT,
):
    CUT_AMOUNT = int(MANUAL_SHIFT * config.BASE_SAMPLE_RATE)
    if CUT_AMOUNT > 0:
        print("Cutting from the beginning of the data idun data")
        idun_base_clipped_data_manual = idun_base_clipped_data[CUT_AMOUNT:]
        idun_clipped_data_manual = idun_clipped_data[CUT_AMOUNT:]

        comparison_base_clipped_df_manual = comparison_base_clipped_df.iloc[
            :-CUT_AMOUNT
        ].reset_index(drop=True)
        comparison_clipped_data_manual = comparison_clipped_data[:-CUT_AMOUNT]

        same_times = np.linspace(
            0,
            len(idun_clipped_data_manual) / config.BASE_SAMPLE_RATE,
            len(idun_clipped_data_manual),
        )
    elif CUT_AMOUNT < 0:
        print("Cutting from the end of the data idun data")
        comparison_clipped_data_manual = comparison_clipped_data[-CUT_AMOUNT:]
        comparison_base_clipped_df_manual = comparison_base_clipped_df.iloc[
            -CUT_AMOUNT:
        ].reset_index(drop=True)
        idun_clipped_data_manual = idun_clipped_data[:CUT_AMOUNT]
        idun_base_clipped_data_manual = idun_base_clipped_data[:CUT_AMOUNT]

        same_times = np.linspace(
            0,
            len(idun_clipped_data_manual) / config.BASE_SAMPLE_RATE,
            len(idun_clipped_data_manual),
        )
    else:
        print("No cutting")
        idun_base_clipped_data_manual = copy.deepcopy(idun_base_clipped_data)
        comparison_base_clipped_df_manual = copy.deepcopy(comparison_base_clipped_df)
        comparison_clipped_data_manual = copy.deepcopy(comparison_clipped_data)
        idun_clipped_data_manual = copy.deepcopy(idun_clipped_data)
        same_times = np.linspace(
            0,
            len(idun_clipped_data_manual) / config.BASE_SAMPLE_RATE,
            len(idun_clipped_data_manual),
        )
    return (
        comparison_clipped_data_manual,
        idun_clipped_data_manual,
        idun_base_clipped_data_manual,
        comparison_base_clipped_df_manual,
        same_times,
    )


def load_edf_file(folder, subject, night, original_sample_rate):
    """
    Load edf file from the folder

    Args:
        folder (str): folder name
        subject (str): subject name
        night (str): night name

    Returns:
        list: list of raw data and channel names
    """
    edf_file_path = glob.glob(os.path.join(folder, subject, night, "*scoring.edf"))[0]
    complete_edf_file = highlevel.read_edf(edf_file_path)
    edf_file_data = complete_edf_file[0]
    edf_file_chan = complete_edf_file[1]
    target_length = len(complete_edf_file[0][1])
    raw_data = []
    channel_names = []
    # resample all channels that did not have 120Hz sampling Freq
    for chan_indx in range(len(edf_file_data)):
        samp_freq = edf_file_chan[chan_indx]["sample_frequency"]
        chan_unit = edf_file_chan[chan_indx]["dimension"]
        channel_data = pyedflib_to_mne(
            edf_file_data[chan_indx],
            edf_file_chan[chan_indx],
            target_length,
            samp_freq,
            chan_unit,
        )
        raw_data.append(channel_data)
        channel_names.append(edf_file_chan[chan_indx]["label"])
    # convert list to np array
    raw_data = np.vstack(raw_data)
    comparison_raw_data = []
    comparison_raw_data.append(raw_data)
    comparison_raw_data.append(channel_names)
    comparison_time_stamps = create_timestamp_array(target_length, original_sample_rate)
    file_extention = "edf"
    return comparison_raw_data, comparison_time_stamps, file_extention


def load_xdf_file(folder, subject, night):
    xdf_file_path = glob.glob(os.path.join(folder, subject, night, "*.xdf"))[0]
    print(xdf_file_path)
    comparison_raw_data, _ = pyxdf.load_xdf(xdf_file_path)
    comparison_time_stamps = comparison_raw_data[0]["time_stamps"]
    file_extention = "xdf"
    return comparison_raw_data, comparison_time_stamps, file_extention


def manual_sync_automatic(
    comparison_clipped_data,
    idun_clipped_data,
    idun_base_clipped_data,
    comparison_base_clipped_df,
    shift,
):
    CUT_AMOUNT = int(shift * config.BASE_SAMPLE_RATE)
    if CUT_AMOUNT > 0:
        print("Cutting from the beginning of the data idun data")
        idun_base_clipped_data_manual = idun_base_clipped_data[CUT_AMOUNT:]
        idun_clipped_data_manual = idun_clipped_data[CUT_AMOUNT:]

        comparison_base_clipped_df_manual = comparison_base_clipped_df.iloc[
            :-CUT_AMOUNT
        ].reset_index(drop=True)
        comparison_clipped_data_manual = comparison_clipped_data[:-CUT_AMOUNT]

        same_times = np.linspace(
            0,
            len(idun_clipped_data_manual) / config.BASE_SAMPLE_RATE,
            len(idun_clipped_data_manual),
        )
    elif CUT_AMOUNT < 0:
        print("Cutting from the end of the data idun data")
        comparison_clipped_data_manual = comparison_clipped_data[-CUT_AMOUNT:]
        comparison_base_clipped_df_manual = comparison_base_clipped_df.iloc[
            -CUT_AMOUNT:
        ].reset_index(drop=True)
        idun_clipped_data_manual = idun_clipped_data[:CUT_AMOUNT]
        idun_base_clipped_data_manual = idun_base_clipped_data[:CUT_AMOUNT]

        same_times = np.linspace(
            0,
            len(idun_clipped_data_manual) / config.BASE_SAMPLE_RATE,
            len(idun_clipped_data_manual),
        )
    else:
        print("No cutting")
        idun_base_clipped_data_manual = copy.deepcopy(idun_base_clipped_data)
        comparison_base_clipped_df_manual = copy.deepcopy(comparison_base_clipped_df)
        comparison_clipped_data_manual = copy.deepcopy(comparison_clipped_data)
        idun_clipped_data_manual = copy.deepcopy(idun_clipped_data)
        same_times = np.linspace(
            0,
            len(idun_clipped_data_manual) / config.BASE_SAMPLE_RATE,
            len(idun_clipped_data_manual),
        )
    return (
        comparison_clipped_data_manual,
        idun_clipped_data_manual,
        idun_base_clipped_data_manual,
        comparison_base_clipped_df_manual,
        same_times,
    )

def plot_sync_results(lag_arr_copy,shift,config):
    plt.figure(figsize=(15, 2))
    plot_time_arr = np.linspace(
        0, len(lag_arr_copy) * config.FIRST_LAG_EPOCH_SIZE, len(lag_arr_copy)
    )
    # convert  to seconds
    plot_time_arr = plot_time_arr / config.BASE_SAMPLE_RATE
    plt.title(f"Lag over time with shift: {shift}")
    plt.xlabel("Time (s)")
    plt.ylabel("Lag (s)")
    plt.plot(plot_time_arr, np.array(lag_arr_copy) / config.BASE_SAMPLE_RATE)
    plt.show()
    
def custom_sort(arr):
    arr.sort(key=lambda x: (abs(x), -x))
    return arr[::-1]

def find_automatic_alignment(
    comparison_clipped_data,
    idun_clipped_data,
    idun_base_clipped_data,
    comparison_base_clipped_df,
    config
    ):
    
    SHIFT_SIGN = [-60, 60]
    SHIFTS = list(range(SHIFT_SIGN[0], SHIFT_SIGN[1], 20))
    SHIFTS = custom_sort(SHIFTS)
    SYNCED_LOSS_THRESH = 0.5

    for shift in SHIFTS:
        print("\n------------------------")
        print(f"Testing shift of: {shift}s")
        print("------------------------")
        (
            comparison_clipped_data_manual,
            idun_clipped_data_manual,
            idun_base_clipped_data_manual,
            comparison_base_clipped_df_manual,
            same_times,
        ) = manual_sync_automatic(
            comparison_clipped_data,
            idun_clipped_data,
            idun_base_clipped_data,
            comparison_base_clipped_df,
            shift,
        )
        comparison_clipped_temp_data = copy.deepcopy(comparison_clipped_data_manual)
        idun_clipped_temp_data = copy.deepcopy(idun_clipped_data_manual)

        search_size = config.FIRST_LAG_EPOCH_SIZE

        comparison_epochs = epoch_data(comparison_clipped_temp_data, search_size)
        idun_epochs = epoch_data(idun_clipped_temp_data, search_size)

        _, _, lag_arr = calculate_epochs_lag(
            comparison_epochs, idun_epochs
        )
        lag_arr_copy = lag_arr[1:].copy()

        plot_sync_results(lag_arr_copy,shift,config)
        cleaned_fine_lag_arr = clean_data_from_spikes(
                lag_arr_copy, config.DISCONTINUITY_THRESHOLD
            )

        cleaned_fine_lag_arr_test = cleaned_fine_lag_arr[
            ~np.isnan(cleaned_fine_lag_arr)
        ]
        new_len = len(cleaned_fine_lag_arr_test)
        old_len = len(lag_arr_copy)
        print(f"New len: {new_len}, Old len: {old_len}")

        if new_len > SYNCED_LOSS_THRESH * old_len:
            print(f"Shift is satisfactory of amount: {shift}s")
            break
    
    return (
        comparison_clipped_data_manual,
        idun_clipped_data_manual,
        idun_base_clipped_data_manual,
        comparison_base_clipped_df_manual,
        same_times,
    )
