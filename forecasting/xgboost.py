import numpy as np
import pandas as pd
import xgboost as xgb

class Xgboost:
    def __init__(self, config_loader, quantiles=None):
        self.config_loader = config_loader
        self.alpha = self.config_loader.get_alpha()
        self.params = self.config_loader.get_params()
        self.param_dist = self.config_loader.get_param_dist()
        self.quantiles = quantiles
        self.models = {}
        self._configure_params()

    def _configure_params(self):
        if self.quantiles is not None:
            for quantile in self.quantiles:
                quantile_params = self.params.copy()
                quantile_params['objective'] = 'reg:quantileerror'
                quantile_params['quantile'] = quantile
                self.models[quantile] = {
                    'params': quantile_params,
                    'model': None
                }

    def fit(self, X, y):
        if self.quantiles is not None:
            for quantile in self.quantiles:
                self.models[quantile]['params']['base_score'] = y.mean()
                self.param_dist['base_score'] = np.arange(y.mean() - 5000, y.mean() + 5000, 500)
                self.models[quantile]['model'] = xgb.XGBRegressor(**self.models[quantile]['params'])
                self.models[quantile]['model'].fit(X, y)
                print(f"Model for quantile {quantile} fitted with parameters:", self.models[quantile]['params'])
        else:
            self.params['base_score'] = y.mean()
            self.param_dist['base_score'] = np.arange(y.mean() - 5000, y.mean() + 5000, 500)
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X, y)
            print("Model fitted with parameters:", self.params)

    def predict(self, X):
        predictions = {}
        if self.quantiles is not None:
            for quantile in self.quantiles:
                if self.models[quantile]['model'] is not None:
                    predictions[quantile] = self.models[quantile]['model'].predict(X)
                    print(f"Predictions for quantile {quantile} computed.")
                else:
                    raise ValueError(f"Model for quantile {quantile} is not fitted yet. Please call fit() before predict().")
        else:
            if self.model is not None:
                predictions = self.model.predict(X)
                print("Predictions computed.")
            else:
                raise ValueError("Model is not fitted yet. Please call fit() before predict().")
        return predictions

    def create_prediction_dataframe(self, predictions, y_test):
        try:
            data = {f'Prediction_{q}': predictions[q] for q in self.quantiles}
            data['Consumption'] = y_test.values
            return pd.DataFrame(data, index=y_test.index)
        except Exception as e:
            print(f"An error occurred during the creation of the prediction dataframe: {e}")
            raise



