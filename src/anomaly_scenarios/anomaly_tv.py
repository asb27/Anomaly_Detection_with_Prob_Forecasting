import pandas as pd
from src.anomaly_scenarios.anomaly_base import AnomalyBase

class AnomalyTv(AnomalyBase):
    def __init__(self, config_loader):
        super().__init__(config_loader)

    def apply_anomaly(self, df, period):
        start_time = pd.to_datetime(period[0])
        end_time = pd.to_datetime(period[1])
        total_period = (end_time - start_time).total_seconds() / 3600  # saat cinsinden

        decrease_percentages = [
            (0, 12.5, 0.9),
            (12.5, 25, 0.75),
            (25, 37.5, 0.7),
            (37.5, 50, 0.65),
            (50, 62.5, 0.6),
        ]

        increase_percentages = [
            (62.5, 68.75, 2.5),
            (68.75, 75, 2.4),
            (75, 81.25, 2.3),
            (81.25, 87.5, 2.2),
            (87.5, 93.75, 2.1),
            (93.75, 100, 2.0),
        ]

        # Update Consumption values for the anomaly period
        for start_percent, end_percent, mult in decrease_percentages:
            period_start = start_time + pd.Timedelta(hours=total_period * start_percent / 100)
            period_end = start_time + pd.Timedelta(hours=total_period * end_percent / 100)
            mask = (df.index >= period_start) & (df.index < period_end)
            df.loc[mask, 'Anomaly_Consumption'] *= mult
            df.loc[mask, 'Anomaly'] = 1
            df.loc[mask, 'Scenario'] = 'TV'

        for start_percent, end_percent, mult in increase_percentages:
            period_start = start_time + pd.Timedelta(hours=total_period * start_percent / 100)
            period_end = start_time + pd.Timedelta(hours=total_period * end_percent / 100)
            mask = (df.index >= period_start) & (df.index < period_end)
            df.loc[mask, 'Anomaly_Consumption'] *= mult
            df.loc[mask, 'Anomaly'] = 1
            df.loc[mask, 'Scenario'] = 'TV'


        return df
