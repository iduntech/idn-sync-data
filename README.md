## Data Stream Synchronization Script

---

### Overview:

This script is designed to synchronize two data streams, Prodigy and IDUN. It works by first unpacking the raw data from these sources, followed by making the datasets equal in length. Subsequent steps involve manual and automated adjustments to align the two datasets.

---

### Key Steps:

1. **Data Acquisition**: Read raw data files (in EDF and CSV formats) for the specified subject and night.

2. **Unpacking Data**:
    - Extract Prodigy data and perform FFT preparation.
    - Extract IDUN data.

3. **Data Length Equalization**: Ensure that Prodigy and IDUN data have the same length.

4. **Data Shifting**:
    - Apply an initial shift to align data streams.
    - Calculate lag between data streams.
    - Cut data based on initial lag analysis.

5. **Data Synchronization**:
    - Perform fine-grained synchronization.
    - Clean lag estimations and fit a polynomial regression.
    - Use linear regression to estimate the difference in sampling rates.
    - Cut the data throughout and at the end based on the estimated difference.

6. **Final Validation and Visualization**:
    - Adjust data using the final mean lag.
    - Plot the synchronized datasets for verification.

---

### Required Libraries:

- `glob`
- `os`
- `numpy as np`
- `mne`
- `copy`

### Configuration:

Ensure that `config` is properly set up with necessary configurations such as `FILTER_RANGE`, `BASE_SAMPLE_RATE`, `FIRST_LAG_EPOCH_SIZE`, and other parameters.

---

### Usage:

1. Ensure your working directory contains the folder structure as: `01_Pre_study/S005/night3`.
2. Ensure that the `config` variable has been set up appropriately.
3. Modify the `SHIFT_SECONDS` value if needed.
4. Modify other parameters like `DISCONTINUITY_THRESHOLD` and `POLYNOMIAL_DEGREE` as necessary.
5. Run the script to process the data and synchronize the two streams.
6. View the final synchronized and filtered data.

---

### Important:

- The script includes sections for manual shifting. Ensure to review and adjust as necessary.
- Always back up raw data before applying any modifications.
- It is advisable to verify the synchronized data by visual inspection to ensure that the algorithm has aligned the two data streams accurately.

---

### Feedback and Contributions:

Feedback and contributions are welcome. If you find any issues or potential improvements, please open an issue or submit a pull request.

--- 

Note: This README provides a general overview and guide for using the provided script. Ensure to understand each step of the script before using it on critical data.