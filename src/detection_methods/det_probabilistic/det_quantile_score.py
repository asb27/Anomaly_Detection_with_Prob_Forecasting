import numpy as np
import pandas as pd
from statistics import mean
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class DetQuantileScore:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def get_score(self, value, quantile_values):
        if value < quantile_values[0]:  # value < 0.0001
            return 21

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


    def normalize(self, series, min_value, max_value):
        return 100 * (series - min_value) / (max_value - min_value)

    def detect(self, df):
        quantiles = self.parameters['quantiles']
        av_score_threshold = self.parameters['average_score']
        an_score_threshold = self.parameters['anomaly_score']
        sequence_number = self.parameters['sequence_number']

        weight_quantile = self.parameters.get('weight_quantile', 0.65)
        weight_diff = self.parameters.get('weight_diff', 0.0)
        weight_residual = self.parameters.get('weight_residual', 0.35)

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

        # Quantile Score
        anomaly_scores = []
        controller = False  # Anomaly controller
        for i in range(len(df)):
            value = df.loc[df.index[i], consumption_column]
            quantile_values = [df.iloc[i, idx] for idx in quantile_column_indices]

            score = self.get_score(value, quantile_values)
            anomaly_scores.append(score)

            df.loc[df.index[i], individual_score_column] = score



        df[diff_score_column] = df[consumption_column].diff().fillna(0).abs()

        model = ExponentialSmoothing(df[consumption_column], trend=None, seasonal=None, seasonal_periods=1)
        fitted_model = model.fit(smoothing_level=0.1, optimized=False)
        df['Smoothed'] = fitted_model.fittedvalues
        df[residuals_column] = (df[consumption_column] - df['Smoothed']).abs()

        df[individual_score_column] = self.normalize(df[individual_score_column], df[individual_score_column].min(),
                                                     df[individual_score_column].max())
        df[diff_score_column] = self.normalize(df[diff_score_column], df[diff_score_column].min(),
                                               df[diff_score_column].max())
        df[residuals_column] = self.normalize(df[residuals_column], df[residuals_column].min(),
                                              df[residuals_column].max())

        for i in range(len(df)):
            weighted_score = (
                    weight_quantile * df.loc[df.index[i], individual_score_column] +
                    weight_diff * df.loc[df.index[i], diff_score_column] +
                    weight_residual * df.loc[df.index[i], residuals_column]
            )
            df.loc[df.index[i], anomaly_score_column] = weighted_score

            if i >= sequence_number - 1:
                past_scores = df[anomaly_score_column].iloc[i - sequence_number + 1:i + 1]
                avg_score = past_scores.mean()
                df.loc[df.index[i], average_score_column] = avg_score
                average_scores = df.loc[df.index[i - sequence_number + 1:i + 1], average_score_column]
                anomaly_scores = df.loc[df.index[i - sequence_number + 1:i + 1], anomaly_score_column]
                anomaly_consumption = df.loc[df.index[i - sequence_number + 1:i + 1], consumption_column]

                if( all(score > av_score_threshold for score in average_scores)
                        and all(score > an_score_threshold for score in anomaly_scores) \
                        and
                        all( df.loc[df.index[i], "Prediction_0.001"] >an  or
                             (an > df.loc[df.index[i], "Prediction_0.99"])
                                for an in anomaly_consumption
                            )
                ):
                        df.loc[df.index[i], detect_column_name] = 1
        

        return df

