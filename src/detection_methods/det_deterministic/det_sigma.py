import numpy as np

class DetSigmaThreshold:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def detect(self, df):
        sigma_level = self.parameters.get("sigma", 3)
        consumption_column = 'Anomaly_Consumption'

        if consumption_column not in df.columns:
            raise ValueError(f"Column {consumption_column} not found in DataFrame")

        detect_column_name = 'Detect'

        df = df.copy()

        # calculate mean and std deviation
        mean = df[consumption_column].mean()
        std_dev = df[consumption_column].std()

        #set lower and upper sigma thresholds
        lower_threshold = mean - sigma_level * std_dev
        upper_threshold = mean + sigma_level * std_dev

        # mark the anomalies
        df.loc[:, detect_column_name] = (
            (df[consumption_column] < lower_threshold) |
            (df[consumption_column] > upper_threshold)
        ).astype(int)

        return df
