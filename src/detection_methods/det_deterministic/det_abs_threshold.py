from sklearn.metrics import mean_absolute_error


class DetAbsThreshold:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def detect(self, df):
        prediction_column = df["Prediction"]
        coefficient = self.parameters["coefficient"]
        consumption_column = 'Anomaly_Consumption'
        mae = mean_absolute_error(df["Consumption"], df["Prediction"])
        print(f"Mean Absolute Error for deterministic deection: {mae}")
        if "Prediction" not in df.columns:
            raise ValueError(f"Columns {"Prediction"}  not found in DataFrame")

        detect_column_name = 'Detect'

        df = df.copy()

        lower_threshold = prediction_column - (coefficient*mae)
        upper_threshold = prediction_column + (coefficient*mae)

        df.loc[:, detect_column_name] = (
                (df[consumption_column] < lower_threshold) |
                (df[consumption_column] > upper_threshold)
        ).astype(int)

        return df
