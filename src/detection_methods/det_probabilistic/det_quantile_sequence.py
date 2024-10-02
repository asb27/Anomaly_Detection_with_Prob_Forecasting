class DetQuantileSequence:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def detect(self, df):
        upper_column = f'Prediction_{self.parameters["upper_quantile"]}'
        lower_column = f'Prediction_{self.parameters["lower_quantile"]}'
        sequence_number = self.parameters["sequence_number"]
        consumption_column = 'Anomaly_Consumption'

        if upper_column not in df.columns or lower_column not in df.columns:
            raise ValueError(f"Columns {upper_column} or {lower_column} not found in DataFrame")

        df = df.copy()

        detect_column_name = 'Detect'
        df[detect_column_name] = 0

        anomaly_mask = (
            (df[consumption_column] < df[lower_column]) |
            (df[consumption_column] > df[upper_column])
        ).astype(int)

        # Anomalies are marked if the anomaly sequence is greater than or equal to the sequence number
        df[detect_column_name] = (
            anomaly_mask.rolling(window=sequence_number, min_periods=1)
                        .sum() >= sequence_number
        ).astype(int)

        return df
