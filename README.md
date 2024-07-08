# Anomaly Detection with Probabilistic Forecasting

This project uses various forecasting and detection methods to identify anomalies in household energy consumption data.

## Contents

### data_preparation/

Contains classes related to data preparation and processing.

### forecasting/

Contains classes for building and applying probabilistic forecasting models.

### anomaly_scenarios/

Contains classes representing different anomaly scenarios and normal data.

### detection_methods/

Contains classes representing different anomaly detection methods.

### evaluation/

Contains the class that evaluates the performance of detection methods.


### Main File: main.py

The main file brings together all modules and executes data processing, forecasting, anomaly application, detection, and evaluation processes. 
But you need data from https://www.nature.com/articles/s41597-022-01156-1.

### Helper File: load_model_df_pkl.py

Loads previously saved forecast data (`prepared_data.pkl`) and quantile model (`quantile_model.pkl`), applies anomaly scenarios, and runs detection methods.


## Usage

The project operates based on a configuration file (`config.json`). This file contains settings for data processing, forecasting, anomaly scenarios, and detection methods. The main file (`main.py`) uses these settings to execute the processes sequentially.

### Setup and Execution

1. Run the main file to train the model and evaluate the results:
    ```bash
    python main.py
    ```

2. To run detection methods with previously trained model and data:
    ```bash
    python load_model_df_pkl.py
    ```

### Configuration File: config.json

The `config.json` file contains settings for data processing, forecasting, anomaly scenarios, and detection methods. This file is used to configure the project.

