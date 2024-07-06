from data_preparation import data_processor
from forecasting.quantile_regression import QuantileRegression

class ModelFactory:
    @staticmethod
    def get_forecasting_method(config_loader):
        model_type = config_loader.get_forecasting_method()
        if model_type == "quantile_regression":
            return QuantileRegression(config_loader,data_processor)

        #elif model_type == "xgboost":
        #    return xgboost(config_loader)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")
