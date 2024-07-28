import pandas as pd

class AnomalyBase:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.anomaly_config = config_loader.config['anomalies']
        self.anomaly_types = self.anomaly_config['types']
        self.anomaly_periods = self.anomaly_config['periods']

    def apply_anomalies(self, all_dfs, anomaly_classes):
        if not isinstance(all_dfs, dict):
            raise ValueError("Input should be a dictionary of DataFrames.")

        results = {}

        for key, df in all_dfs.items():
            if df is None:
                raise ValueError(f"DataFrame for key {key} is None. Please provide a valid DataFrame.")

            df_original = df.copy()
            df['Anomaly_Consumption'] = df['Consumption'].copy()
            df['Anomaly'] = 0
            df['Scenario'] = 'Normal'

            all_anomaly_periods = []

            for anomaly_type in self.anomaly_types:
                anomaly_instance = anomaly_classes[anomaly_type](self.config_loader)
                for period in self.anomaly_periods[anomaly_type]:
                    start_time = pd.to_datetime(period[0]) - pd.Timedelta(hours=4)
                    end_time = pd.to_datetime(period[1]) + pd.Timedelta(hours=4)
                    all_anomaly_periods.append((start_time, end_time))
                    mask = (df.index >= start_time) & (df.index <= end_time)
                    df_period = df.loc[mask].copy()

                    df_updated_period = anomaly_instance.apply_anomaly(df_period, period)

                    df.update(df_updated_period)

            filtered_df = self._filter_anomaly_periods(df, all_anomaly_periods)

            results[key] = (filtered_df, df_original)

        return results

    def apply_anomaly(self, df, period):
        raise NotImplementedError("apply_anomaly method must be implemented in child classes")

    def _filter_anomaly_periods(self, df, anomaly_periods):
        masks = []
        for start_time, end_time in anomaly_periods:
            mask = (df.index >= start_time) & (df.index <= end_time)
            masks.append(mask)
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask |= mask
        return df.loc[combined_mask]
