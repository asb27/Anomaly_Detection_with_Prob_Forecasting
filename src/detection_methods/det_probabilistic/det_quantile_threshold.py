class DetQuantileThreshold:
    def __init__(self, config_loader, params):
            self.config_loader = config_loader
            self.parameters = params


    def detect(self, df):
        upper_column = f'Prediction_{self.parameters["upper_quantile"]}'
        lower_column = f'Prediction_{self.parameters["lower_quantile"]}'
        threshold = self.parameters["threshold"]
        consumption_column = 'Anomaly_Consumption'

        if upper_column not in df.columns or lower_column not in df.columns:
            raise ValueError(f"Columns {upper_column} or {lower_column} not found in DataFrame")

        detect_column_name = 'Detect'

        #df = df.copy()

        lower_threshold = df[lower_column] * (1 - threshold)
        upper_threshold = df[upper_column] * (1 + threshold)
        df.loc[:, detect_column_name] = (
                (df[consumption_column] < lower_threshold) |
                (df[consumption_column] > upper_threshold)
        ).astype(int)

        return df
