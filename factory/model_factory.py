from data_preparation import data_processor
from forecasting.quantile_regression import QuantileRegression
from forecasting.xgboost import Xgboost
class ModelFactory:


    def __init__(self, config_loader):
        self.config_loader = config_loader

    @staticmethod
    def get_forecasting_method(forecasting_type, config_loader):
        model_type = forecasting_type
        if model_type == "quantile_regression":
            return QuantileRegression(config_loader)
        elif model_type == "xgboost":
            return Xgboost(config_loader)

        #elif model_type == "xgboost":
        #    return xgboost(config_loader)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")
