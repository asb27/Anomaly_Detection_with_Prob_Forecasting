import pandas as pd
import numpy as np
from src.anomaly_scenarios.anomaly_base import AnomalyBase

class AnomalyHeatpump(AnomalyBase):
    def __init__(self, config_loader):
        super().__init__(config_loader)

    def apply_anomaly(self, df, period):
        start_time = pd.to_datetime(period[0])
        end_time = pd.to_datetime(period[1])

        # Calculate total hours in the period
        total_period = (end_time - start_time).total_seconds() / 3600

        # Get original consumption value at the start
        original_start_value = df.loc[start_time, 'Anomaly_Consumption']

        # Define the key points for modifications
        original_values = [
            original_start_value * 3,
            original_start_value * 2.9,
            original_start_value * 2.9,
            original_start_value * 2.8,
            original_start_value * 1.9,

        ]
        original_times = [0, 40, 60, 80, 100]

        new_values = []
        new_times = []

        for i in range(len(original_values)):
            if i < len(original_values) - 1:
                mid_value1 = (original_values[i] + original_values[i + 1]) * 0.48
                mid_value2 = (original_values[i] + original_values[i + 1]) * 0.53
                mid_time1 = original_times[i] + (original_times[i + 1] - original_times[i]) * 0.47
                mid_time2 = original_times[i] + (original_times[i + 1] - original_times[i]) * 0.55
                new_values.extend([original_values[i], mid_value1, mid_value2])
                new_times.extend([original_times[i], mid_time1, mid_time2])
            else:
                new_values.append(original_values[i])
                new_times.append(original_times[i])

        time_points = [start_time + pd.Timedelta(hours=total_period * t / 100) for t in new_times]

        rounded_time_points = [df.index.get_loc(df.index[df.index.get_indexer([tp], method='nearest')[0]]) for tp in time_points]

        for i in range(len(time_points)):
            df.loc[df.index[rounded_time_points[i]], 'Anomaly_Consumption'] = new_values[i]

        for i in range(len(time_points) - 1):
            mask = (df.index > df.index[rounded_time_points[i]]) & (df.index < df.index[rounded_time_points[i + 1]])
            df.loc[mask, 'Anomaly_Consumption'] = np.nan

        df['Anomaly_Consumption'] = df['Anomaly_Consumption'].interpolate(method='linear')

        full_mask = (df.index >= start_time) & (df.index <= end_time)
        df.loc[full_mask, 'Anomaly'] = 1
        df.loc[full_mask, 'Scenario'] = 'Heatpump'

        return df
