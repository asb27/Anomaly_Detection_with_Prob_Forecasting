from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing


class DETSmoothing:
    """
    This class detect Anomalies  using  smoothing  score and Anomaly Score values.
    """
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def get_score(self, value, quantile_values):
        if value < quantile_values[0]:  # value < 0.0001
            return 23

        for i in range(len(quantile_values) - 1):
            if quantile_values[i] <= value < quantile_values[i + 1]:
                if i == 0:  # 0.0001 < value < 0.0005
                    return 15
                elif i == 1:  # 0.0005 < value < 0.001
                    return 10
                elif i == 2:  # 0.001 < value < 0.005
                    return 6
                elif i == 3:  # 0.005 < value < 0.01
                    return 5
                elif i == 4:  # 0.01 < value < 0.05
                    return 2
                elif i == 5:  # 0.05 < value < 0.1
                    return 1
                elif 6 <= i <= 13:  # 0.1 < value < 0.9
                    return 0
                elif i == 14:  # 0.9 < value < 0.95
                    return 1
                elif i == 15:  # 0.95 < value < 0.99
                    return 3
                elif i == 16:  # 0.99 < value < 0.995
                    return 8
                elif i == 17:  # 0.995 < value < 0.999
                    return 22
                elif i == 18:  # 0.999 < value < 0.9995
                    return 40
                elif i == 19:  # 0.9995 < value < 0.9999
                    return 60

        if value >= quantile_values[-1]:  # 0.9999 < value
            return 80

    def detect(self, df):
        quantiles = self.parameters['quantiles']
        smoothed_score_threshold = self.parameters['smoothed_score']
        score_threshold = self.parameters['anomaly_score']


        consumption_column = 'Anomaly_Consumption'
        detect_column_name = 'Detect'
        anomaly_score_column = 'Anomaly_Score'
        individual_score_column = 'Individual_Score'
        diff_score_column = 'Diff_Score'
        residuals_column = 'Residuals'
        average_score_column = 'Average_Score'

        df[detect_column_name] = 0
        df[anomaly_score_column] = 0.0
        df[individual_score_column] = 0.0
        df[diff_score_column] = 0.0
        df[residuals_column] = 0.0
        df[average_score_column] = 0.0

        quantile_column_indices = [df.columns.get_loc(f'Prediction_{q}') for q in quantiles]

        anomaly_scores = []
        controller = False
        for i in range(len(df)):
            value = df.loc[df.index[i], consumption_column]
            quantile_values = [df.iloc[i, idx] for idx in quantile_column_indices]

            score = self.get_score(value, quantile_values)
            anomaly_scores.append(score)

            df.loc[df.index[i], individual_score_column] = score


        model = ExponentialSmoothing(df[individual_score_column], trend=None, seasonal=None, seasonal_periods=1)
        fitted_model = model.fit(smoothing_level=0.6, optimized=False)
        df['Smoothed_score'] = fitted_model.fittedvalues

        for i in range(1, len(df)):
            if (df.loc[df.index[i], 'Smoothed_score'] > smoothed_score_threshold and df.loc[df.index[i], individual_score_column] > score_threshold):

                df.loc[df.index[i], detect_column_name] = 1

        return df