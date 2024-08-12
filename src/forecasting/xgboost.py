import pandas as pd
import xgboost as xgb
import numpy as np

class Xgboost:
    def __init__(self, config_loader):
        self.params = config_loader
        print("xgboost_params", self.params)
        self.model = None

    def fit(self, X_train, y_train):
        self.model = xgb.XGBRegressor(**self.params['params'])
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.model is not None:
            return self.model.predict(X_test)
        else:
            print("Model is not fitted yet.")
            return None

    def create_prediction_dataframe(self, predictions, y_test):
        try:
            data = {'Prediction': predictions, 'Consumption': y_test.values}
            return pd.DataFrame(data, index=y_test.index)
        except Exception as e:
            print(f"An error occurred during the creation of the prediction dataframe: {e}")
            raise
