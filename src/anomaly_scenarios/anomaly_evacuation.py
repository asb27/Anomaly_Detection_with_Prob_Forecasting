import pandas as pd
import numpy as np
from src.anomaly_scenarios.anomaly_base import AnomalyBase

class AnomalyEvacuation(AnomalyBase):
    def __init__(self, config_loader):
        super().__init__(config_loader)

    def apply_anomaly(self, df, period):
        start_time = pd.to_datetime(period[0])
        end_time = pd.to_datetime(period[1])

        total_period = (end_time - start_time).total_seconds() / 3600

        original_start_value = df.loc[start_time, 'Anomaly_Consumption']

        original_values = [
            original_start_value * 2.2,
            original_start_value * 3,
            original_start_value * 2.5,
            original_start_value * 0.3,
            original_start_value * 0.2
            ]
        original_times = [0, 40, 60, 65, 100]

        new_values = []
        new_times = []

        for i in range(len(original_values)):
            if i < len(original_values) - 1 and (original_times[i] < 60 or original_times[i] > 65):
                mid_value1 = (original_values[i] + original_values[i + 1]) * 0.49
                mid_value2 = (original_values[i] + original_values[i + 1]) * 0.57
                mid_time1 = original_times[i] + (original_times[i + 1] - original_times[i]) * 0.49
                mid_time2 = original_times[i] + (original_times[i + 1] - original_times[i]) * 0.55
                new_values.extend([original_values[i], mid_value1, mid_value2])
                new_times.extend([original_times[i], mid_time1, mid_time2])
            else:
                new_values.append(original_values[i])
                new_times.append(original_times[i])

        # Convert percentage times to actual times
        time_points = [start_time + pd.Timedelta(hours=total_period * t / 100) for t in new_times]

        # Yuvarlanmış zaman noktalarını bulmak
        rounded_time_points = [df.index.get_loc(df.index[df.index.get_indexer([tp], method='nearest')[0]]) for tp in time_points]

        # Apply the changes and fill in between points using linear interpolation
        for i in range(len(time_points)):
            df.loc[df.index[rounded_time_points[i]], 'Anomaly_Consumption'] = new_values[i]

        # Ensure that the values in between are NaN to allow interpolation
        for i in range(len(time_points) - 1):
            mask = (df.index > df.index[rounded_time_points[i]]) & (df.index < df.index[rounded_time_points[i + 1]])
            df.loc[mask, 'Anomaly_Consumption'] = np.nan

        # Perform the interpolation over the entire period
        df['Anomaly_Consumption'] = df['Anomaly_Consumption'].interpolate(method='linear')

        # Set anomaly flags and scenario name for the entire period
        full_mask = (df.index >= start_time) & (df.index <= end_time)
        df.loc[full_mask, 'Anomaly'] = 1
        df.loc[full_mask, 'Scenario'] = 'Evacuation'

        return df
