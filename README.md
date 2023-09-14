

## EEG Data Processing and Synchronization

### Overview:
This code processes EEG data, syncing data from different sources, and outputs the results as a CSV file. The primary data sources are `.edf` and `.xdf` files, but it also accepts `.csv` inputs.


This script is designed for the synchronization of EEG data from multiple sources. Leveraging the method of cross-correlation, the script quantifies the lag between datasets, ensuring precise alignment. To correct any mismatches, every 30 seconds, one to five samples are removed from the dataset that's leading in time. After processing, the synchronized data from both sources are merged and saved as a `.csv` file.

### Pre-requisites:
Ensure that necessary libraries such as `glob`, `os`, `mne`, `pyxdf`, and `numpy` are imported.

### Steps:

Overview:

1. Configuration: 
    - Specifies the target folder, subject, night, and file type.
  
2. Data Loading: 
    - Checks and loads EEG data from `.edf` and `.xdf` file formats present in the specified directory.
    - Loads corresponding EEG data from a `.csv` file.

3. Data Preprocessing: 
    - Extracts and prepares data from the loaded files, filtering and transforming them for synchronization.

4. Data Synchronization:
    - Equalizes the length of the two data sources.
    - Performs a coarse synchronization, followed by a more fine-grained synchronization.
    - Adjusts for discrepancies in sampling rates.

5. Data Finalization:
    - Makes final adjustments based on calculated lags.
    - Merges synchronized data from both sources into a single DataFrame.

6. Output:
    - Saves the synchronized data to a `.csv` file.

Note: For in-depth details, refer to the comments and markdown cells within the script.
-----------------------------------------------------------------


### Outputs:
1. A CSV file named `{subject}_{night}_synced_data.csv` in the directory defined by the `folder`, `subject`, and `night` parameters.
2. Various visualizations to aid in understanding the processed data.

---

Note: For a comprehensive understanding, one would ideally need to see the complete set of functions that are called within this script (e.g., `get_device_configuration()`, `sync_start_and_equalize_data_length()`, etc.), their implementations, and associated configurations stored in the `config` variable. This README provides a high-level understanding of the code's main flow and operations.