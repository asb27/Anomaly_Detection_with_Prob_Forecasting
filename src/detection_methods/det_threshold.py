class DetThreshold:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def detect(self, df):
        median_column = f'Prediction_{self.parameters["median_quantile"]}'
        threshold = self.parameters["threshold"]
        consumption_column = 'Anomaly_Consumption'

        if median_column not in df.columns:
            raise ValueError(f"Columns {median_column}  not found in DataFrame")

        detect_column_name = 'Detect'

        df = df.copy()

        lower_threshold = df[median_column] * (1 - threshold)
        upper_threshold = df[median_column] * (1 + threshold)

        df.loc[:, detect_column_name] = (
                (df[consumption_column] < lower_threshold) |
                (df[consumption_column] > upper_threshold)
        ).astype(int)

        return df
