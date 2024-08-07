from src.detection_methods.det_quantile import DetQuantile
from src.detection_methods.det_quantile_threshold import DetQuantileThreshold
from src.detection_methods.det_quantile_sequence import DetQuantileSequence
from src.detection_methods.det_quantile_score import DetQuantileScore


class DetectionFactory:
    def __init__(self, config_loader):
        self.config_loader = config_loader

    def get_detection_method(self, method_name, params):
        if method_name == "det_quantile":
            return DetQuantile(self.config_loader, params)
        elif method_name == "det_quantile_threshold":
            return DetQuantileThreshold(self.config_loader, params)
        elif method_name == "det_quantile_sequence":
            return DetQuantileSequence(self.config_loader, params)
        elif method_name == "det_quantile_score":
            return DetQuantileScore(self.config_loader, params)
        else:
            raise ValueError(f"Unknown detection method: {method_name}")
