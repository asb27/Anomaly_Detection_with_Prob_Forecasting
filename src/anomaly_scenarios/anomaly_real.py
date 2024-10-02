import pandas as pd
import numpy as np
from src.anomaly_scenarios.anomaly_base import AnomalyBase

class AnomalyReal(AnomalyBase):
    def __init__(self, config_loader):
        super().__init__(config_loader)

    def apply_anomaly(self, df, period):
        start_time = pd.to_datetime(period[0])
        end_time = pd.to_datetime(period[1])

        mask = (df.index >= start_time) & (df.index <= end_time)
        df.loc[mask, 'Anomaly'] = 1
        df.loc[mask, 'Scenario'] = 'Real'

        return df
