from src.detection_methods.det_quantile import DetQuantile
from src.detection_methods.det_threshold import DetThreshold
from src.detection_methods.det_quantile_threshold import DetQuantileThreshold
from src.detection_methods.det_quantile_sequence import DetQuantileSequence
from src.detection_methods.det_quantile_score import DetQuantileScore
from src.detection_methods.det_cdf_squence import DetCDFSequence
#Deterministic methods
from src.detection_methods.det_deterministic.det_sigma import DetSigmaThreshold



class DetectionFactory:
    def __init__(self, config_loader):
        self.config_loader = config_loader

    def get_detection_method(self, method_name, params):
        if method_name == "det_quantile":
            return DetQuantile(self.config_loader, params)
        elif method_name == "det_threshold":
            return DetThreshold(self.config_loader, params)
        elif method_name == "det_quantile_threshold":
            return DetQuantileThreshold(self.config_loader, params)
        elif method_name == "det_quantile_sequence":
            return DetQuantileSequence(self.config_loader, params)
        elif method_name == "det_quantile_score":
            return DetQuantileScore(self.config_loader, params)
        elif method_name == "det_cdf_sequence":
            return DetCDFSequence(self.config_loader, params)

        #Deterministic methods
        elif method_name == "det_sigma":
            return DetSigmaThreshold(self.config_loader, params)
        else:
            raise ValueError(f"Unknown detection method: {method_name}")
