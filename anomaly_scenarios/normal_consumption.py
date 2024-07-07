import pandas as pd
from anomaly_scenarios.anomaly_base import AnomalyBase

class NormalConsumption(AnomalyBase):
    def __init__(self, config_loader):
        super().__init__(config_loader)

    def apply_anomaly(self, df, period):
        return df