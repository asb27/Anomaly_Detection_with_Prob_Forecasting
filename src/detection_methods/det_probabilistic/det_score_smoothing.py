from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing


class DETScoreSmoothing:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def get_score(self, value, quantile_values):
        if value < quantile_values[0]:  # value < 0.0001
            return 23

        for i in range(len(quantile_values) - 1):
            if quantile_values[i] <= value < quantile_values[i + 1]:
                if 0<i<=6 :  # 0.0001 < value < 0.1
                    return ((7-i)**2)/2

                elif 6 <= i <= 13:  # 0.1 < value < 0.9
                    return 0
                elif 14 < i :  # 0.9 < value
                    return (i-13)**2

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