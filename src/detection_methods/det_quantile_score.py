
class DetQuantileScore:
    def __init__(self, params):
        self.parameters = params
        self.quantiles = self.parameters["quantiles"]

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
        consumption_column = 'Anomaly_Consumption'
        detect_column_name = 'Detect'
        df = df.copy()
        df[detect_column_name] = 0

        quantile_column_indices = [df.columns.get_loc(f'Prediction_{q}') for q in self.quantiles]

        for i in range(len(df)):
            scores = []
            value = df.iloc[i, consumption_column]

            quantile_values = [df.iloc[i, idx] for idx in quantile_column_indices]

            score = self.get_score(value, quantile_values)
            scores.append(score)

            total_score = sum(scores[-self.parameters["sequence_number"]:])

            if total_score > self.parameters['threshold']:
                df.iloc[i, df.columns.get_loc(detect_column_name)] = total_score

        return df

