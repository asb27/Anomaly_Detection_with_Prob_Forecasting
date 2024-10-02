class DetPercentThreshold:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def detect(self, df):
        prediction_column = df["Prediction"]
        threshold = self.parameters["threshold"]
        consumption_column = 'Anomaly_Consumption'

        if "Prediction" not in df.columns:
            raise ValueError(f"Columns {"Prediction"}  not found in DataFrame")

        detect_column_name = 'Detect'

        df = df.copy()

        lower_threshold = prediction_column * (1 - threshold)
        upper_threshold = prediction_column * (1 + threshold)

        df.loc[:, detect_column_name] = (
                (df[consumption_column] < lower_threshold) |
                (df[consumption_column] > upper_threshold)
        ).astype(int)

        return df
