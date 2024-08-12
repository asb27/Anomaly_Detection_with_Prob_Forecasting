import numpy as np
import pandas as pd
from scipy.stats import rankdata

class DetCDFSequence:
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params
        self.lower_bound = params['lower_bound']
        self.upper_bound = params['upper_bound']
        self.sequence_number = params['sequence_number']
    def calculate_cdf_single_row(self, values, target_value):
        sorted_values = np.sort(values)
        rank = np.searchsorted(sorted_values, target_value, side='right')
        cdf_value = rank / len(values)
        return cdf_value

    def detect(self, df):
        quantiles = self.parameters['quantiles']
        quantile_columns = [f'Prediction_{q}' for q in quantiles]

        #quantile_columns = [col for col in df.columns if col.startswith('Prediction_')]

        df['CDF_Anomaly_Consumption'] = 0.0
        detect_column_name = 'Detect'
        df[detect_column_name] = 0

        for index, row in df.iterrows():
            prediction_values = row[quantile_columns].values

            anomaly_value = row['Anomaly_Consumption']

            cdf_value = self.calculate_cdf_single_row(prediction_values, anomaly_value)
            df.loc[index, 'CDF_Anomaly_Consumption'] = cdf_value

            # CDF değerini kontrol et
            if not (self.lower_bound <= cdf_value <= self.upper_bound):
                df.loc[index, detect_column_name] = 1

        return df