from src.anomaly_scenarios.anomaly_heatpump import AnomalyHeatpump
from src.anomaly_scenarios.normal_consumption import NormalConsumption
from src.anomaly_scenarios.anomaly_tv import AnomalyTv
from src.anomaly_scenarios.anomaly_evacuation import AnomalyEvacuation
from src.anomaly_scenarios.anomaly_real import AnomalyReal

class AnomalyFactory:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.classes_config = {
            "normal_consumption": NormalConsumption,
            "anomaly_tv": AnomalyTv,
            "anomaly_evacuation": AnomalyEvacuation,
            "anomaly_heatpump": AnomalyHeatpump,
            "anomaly_real": AnomalyReal
        }

    def get_anomaly_classes(self):
        return self.classes_config
