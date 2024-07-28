from src.detection_methods.detection_with_quantile import DetectionWithQuantile
from src.detection_methods.detection_with_threshold import DetectionWithThreshold

class DetectionFactory:
    def __init__(self, config_loader):
        self.config_loader = config_loader

    def get_detection_method(self, method_name, params):
        if method_name == "det_with_threshold":
            return DetectionWithThreshold(self.config_loader, params)
        elif method_name == "det_with_quantile":
            return DetectionWithQuantile(self.config_loader, params)
        elif method_name == "detection_with_threshold":
            return DetectionWithThreshold(self.config_loader, params)
        else:
            raise ValueError(f"Unknown detection method: {method_name}")
