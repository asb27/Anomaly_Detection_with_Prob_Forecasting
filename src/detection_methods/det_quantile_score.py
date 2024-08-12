class DetQuantileScore:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def get_score(self, value, quantile_values):
        if value < quantile_values[0]:
            return 20

        for i in range(len(quantile_values) - 1):
            if quantile_values[i] <= value < quantile_values[i + 1]:
                if i == 0:
                    return 10
                elif i == 1:
                    return 5
                elif i == 2:
                    return 0
                elif i == 3:
                    return 5
                elif i == 4:
                    return 10

        if value >= quantile_values[-1]:
            return 20

        return 0

    def detect(self, df):
        quantiles = self.parameters['quantiles']
        score_threshold = self.parameters['score']
        sequence_number = self.parameters['sequence_number']

        consumption_column = 'Anomaly_Consumption'
        detect_column_name = 'Detect'
        anomaly_score_column = 'Anomaly_Score'

        df = df.copy()
        df[detect_column_name] = 0
        df[anomaly_score_column] = 0

        quantile_column_indices = [df.columns.get_loc(f'Prediction_{q}') for q in quantiles]
        anomaly_scores = []

        for i in range(len(df)):
            value = df.loc[df.index[i], consumption_column]
            quantile_values = [df.iloc[i, idx] for idx in quantile_column_indices]

            anomaly_score = self.get_score(value, quantile_values)
            anomaly_scores.append(anomaly_score)

            total_score = sum(anomaly_scores[-sequence_number:])

            df.loc[df.index[i], anomaly_score_column] = total_score

            if total_score > score_threshold:
                df.loc[df.index[i], detect_column_name] = 1

        return df
