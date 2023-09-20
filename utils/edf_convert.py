from pyedflib import highlevel
import numpy as np


# function to prepare for edfwriter
def edf_info_all_channels(dataset: np.ndarray, chanlocs: list, sample_freq:250, dimension:'uV'):
    '''
    This function prepares the signal header to save a np array as .edf file if all 
    channels have the same sampling rate and dimension.

    Parameters:
        dataset (numpy array): data to be saved as edf file in format (nr of channels, datapoints)
        chanlocs (list): list of channel names (must match order of signals)
        sampFreq: Sampling Frequency, default = 250

    Returns:
        Signal headers for edf file generation
    '''


    signal_headers = highlevel.make_signal_headers(chanlocs, 
                                               dimension = dimension,
                                               sample_frequency=sample_freq, 
                                               sample_rate=sample_freq,
                                               physical_max = 1,
                                               physical_min = 1,
                                               )
    
    # adjust physical min and max for each channel
    print(chanlocs)
    for n in range(len(chanlocs)):
        signal_headers[n]['physical_min'] = min(dataset[n])
        #print(signal_headers[n]['physical_min'])
        signal_headers[n]['physical_max'] = max(dataset[n])
        #print(signal_headers[n]['physical_max'])

    return signal_headers


# function to individual channels for edfwriter
def edf_info_single_channel(dataset: np.ndarray, chanlocs: list, sample_freq:250, dimension:'uV'):
    '''
    This function prepares the signal header for an individual channel that prepares it for .edf conversion.

    Parameters:
        dataset (numpy array): data to be saved as edf file in format (nr of channels, datapoints)
        chanloc (str): channel name as string 
        sampFreq: Sampling Frequency, default = 250
        dimension: Unit of channel, default = 'uV'

    Returns:
        Signal header for edf file generation
    '''


    signal_header = highlevel.make_signal_header(chanloc, 
                                               dimension = dimension,
                                               sample_frequency=sampFreq, 
                                               sample_rate=sampFreq,
                                               physical_max = 1,
                                               physical_min = 1,
                                               )
    
    # adjust physical min and max for each channel
    signal_header['physical_min'] = min(dataset)
    #print(signal_headers[n]['physical_min'])
    signal_header['physical_max'] = max(dataset)
    #print(signal_headers[n]['physical_max'])

    return signal_header