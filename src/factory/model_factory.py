from src.forecasting.quantile_regression import QuantileRegression
from src.forecasting.qxgboost import QXgboost
from src.forecasting.xgboost import Xgboost
class ModelFactory:


    def __init__(self, config_loader):
        self.config_loader = config_loader

    @staticmethod
    def get_forecasting_method(forecasting_type, config_loader):
        model_type = forecasting_type
        if model_type == "quantile_regression":
            return QuantileRegression(config_loader)
        elif model_type == "qxgboost" or model_type == "qxgboost":
            return QXgboost(config_loader)
        elif model_type == "xgboost" or model_type == "xgboost":
            return Xgboost(config_loader)
        #elif model_type == "arima":
        #    return xgboost(config_loader)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")
