class DetQuantile:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def detect(self, df):
        upper_column = f'Prediction_{self.parameters["upper_quantile"]}'
        lower_column = f'Prediction_{self.parameters["lower_quantile"]}'
        consumption_column = 'Anomaly_Consumption'

        if upper_column not in df.columns or lower_column not in df.columns:
            raise ValueError(f"Columns {upper_column} or {lower_column} not found in DataFrame")

        detect_column_name = 'Detect'

        df = df.copy()  # DataFrame'in bir kopyasını oluşturun

        df.loc[:, detect_column_name] = (
            (df[consumption_column] < df[lower_column]) |
            (df[consumption_column] > df[upper_column])
        ).astype(int)

        return df
