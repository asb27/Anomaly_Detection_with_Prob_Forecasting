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

        anomaly_count = 0

        for i in range(len(df)):

            try:
                is_anomaly = (df.iloc[i][consumption_column] < df.iloc[i][lower_column]) or (
                        df.iloc[i][consumption_column] > df.iloc[i][upper_column]
                )
            except KeyError as e:
                print(f"KeyError at index {i}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error at index {i}: {e}")
                continue

            if is_anomaly:
                anomaly_count += 1
            else:
                anomaly_count = 0

        # If the sequence of anomalies is greater than or equal to the sequence number, mark the anomalies
            if anomaly_count >= sequence_number:
                df.iloc[i - sequence_number + 1:i + 1, df.columns.get_loc(detect_column_name)] = 1

        return df
