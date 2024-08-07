import pandas as pd
from src.factory.detection_factory import DetectionFactory

methods=''
class DetectionBase:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.detection_factory = DetectionFactory(config_loader)
        self.detection_methods = config_loader.get_config('detection_methods')['methods']

    def apply_detection_methods(self, df):
        results = {}
        df = df.copy()
        for method_name in self.detection_methods:
            method_params_list = self.config_loader.get_detection_parameters(method_name)
            #methods = method_name
            for params in method_params_list:
                detection_method = self.detection_factory.get_detection_method(method_name, params)
                detection_result = detection_method.detect(
                    df.copy())

                param_str = '_'.join([f'{key}_{value}' for key, value in params.items()])
                column_name = f'{method_name}_{param_str}'

                #results[column_name] = detection_result['Detect']
                df.loc[:, column_name] = detection_result['Detect'].values
        return df

    def get_anomaly_periods(self, anomaly_type):
        return self.config_loader.get_anomaly_periods(anomaly_type)

    def get_one_day_anomalies(self, df, anomaly_periods):
        anomaly_start = pd.to_datetime(anomaly_periods[0][0])
        anomaly_end = pd.to_datetime(anomaly_periods[0][1])
        return df.loc[
            (df.index >= (anomaly_start - pd.Timedelta(days=1))) &
            (df.index <= (anomaly_end + pd.Timedelta(days=1)))
            ].copy()


    def all_detection_methods(self, all_dfs):
        if not isinstance(all_dfs, dict):
            raise ValueError("Input should be a dictionary of DataFrames.")

        results = {}

        for key, df in all_dfs.items():
            if df is None:
                raise ValueError(f"DataFrame for key {key} is None. Please provide a valid DataFrame.")

            print(f"Applying detection methods to {key}")

            detection_results = self.apply_detection_methods(df)

            results[key] = detection_results

        return results

