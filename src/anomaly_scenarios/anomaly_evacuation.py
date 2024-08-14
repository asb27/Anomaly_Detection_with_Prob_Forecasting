import pandas as pd
from src.anomaly_scenarios.anomaly_base import AnomalyBase


class AnomalyEvacuation(AnomalyBase):
    def __init__(self, config_loader):
        super().__init__(config_loader)
        #self.parameters = config_loader.get_anomaly_periods('anomaly_tv')


    def apply_anomaly(self, df, period):


        start_time = pd.to_datetime(period[0])
        end_time = pd.to_datetime(period[1])
        #mask = (df.index >= start_time) & (df.index <= end_time)


        total_period = (end_time - start_time).total_seconds() / 3600

        increase_percentages = [
            (0, 12.5, 2),
            (12.5, 25, 2.5),
            (25, 37.5, 2.8),
            (37.5, 50, 3.2),
            (50, 62.5, 2.1),
        ]

        # Azalma yüzdeleri (sonra azalma olacak)
        decrease_percentages = [
            (62.5, 68.75, 0.9),
            (68.75, 75, 0.5),
            (75, 81.25, 0.4),
            (81.25, 87.5, 0.3),
            (87.5, 93.75, 0.3),
            (93.75, 100, 0.1),
        ]

        # Update the consumption values for the anomaly period
        for start_percent, end_percent, mult in increase_percentages:
            period_start = start_time + pd.Timedelta(hours=total_period * start_percent / 100)
            period_end = start_time + pd.Timedelta(hours=total_period * end_percent / 100)
            mask = (df.index >= period_start) & (df.index < period_end)
            df.loc[mask, 'Anomaly_Consumption'] *= mult
            df.loc[mask, 'Anomaly'] = 1
            df.loc[mask, 'Scenario'] = 'Evacuation'

        for start_percent, end_percent, mult in decrease_percentages:
            period_start = start_time + pd.Timedelta(hours=total_period * start_percent / 100)
            period_end = start_time + pd.Timedelta(hours=total_period * end_percent / 100)
            mask = (df.index >= period_start) & (df.index < period_end)
            df.loc[mask, 'Anomaly_Consumption'] *= mult
            df.loc[mask, 'Anomaly'] = 1
            df.loc[mask, 'Scenario'] = 'Evacuation'

        return df
