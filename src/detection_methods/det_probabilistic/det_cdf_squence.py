import numpy as np
import pandas as pd
from scipy.stats import rankdata

class DetCDFSequence:
    """
    This class is used to detect anomalies by using the CDF values without any assumed Distribution.
    """
    def __init__(self, config_loader, params):
        self.config_loader = config_loader
        self.parameters = params

    def calculate_cdf(self, df):
        quantiles = self.parameters['quantiles']
        consumption_column = 'Anomaly_Consumption'

        cdf_values = []

        for i in range(len(df)):
            value = df.loc[df.index[i], consumption_column]
            quantile_values = np.array([df.loc[df.index[i], f'Prediction_{q}'] for q in quantiles])

            cdf = np.searchsorted(quantile_values, value, side='right') / len(quantile_values)
            cdf_values.append(cdf)

        df['CDF_Value'] = cdf_values
        return df

    def detect(self, df):
        df = self.calculate_cdf(df)
        detect_column_name = 'Detect'

        df[detect_column_name] = 0

        seq_num = self.parameters.get('sequence_number', 3)  # Varsayılan olarak 3 alır, eğer parametrelerde belirtilmemişse
        lower_bound = self.parameters.get('lower_bound', 0.05)
        upper_bound = self.parameters.get('upper_bound', 0.81)

        for i in range(len(df) - seq_num + 1):
            cdf_values = df.loc[df.index[i:i + seq_num], 'CDF_Value'].tolist()

            if all(cdf_value < lower_bound or cdf_value > upper_bound for cdf_value in cdf_values):
                df.loc[df.index[i], detect_column_name] = 1

        return df
