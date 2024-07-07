from anomaly_scenarios.normal_consumption import NormalConsumption
from anomaly_scenarios.anomaly_tv import AnomalyTv
from anomaly_scenarios.anomaly_evacuation import AnomalyEvacuation

class AnomalyFactory:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.classes_config = {
            "normal_consumption": NormalConsumption,
            "anomaly_tv": AnomalyTv,
            "anomaly_evacuation": AnomalyEvacuation,
        }

    def get_anomaly_classes(self):
        return self.classes_config
