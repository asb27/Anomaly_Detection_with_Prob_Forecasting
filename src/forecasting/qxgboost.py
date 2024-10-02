import pandas as pd
import xgboost as xgb
import numpy as np

class QXgboost:
    def __init__(self, config_loader):
        self.params = config_loader
        self.alpha = self.params.get('params', {}).get('quantile_alpha', [0.1, 0.5,0.9])
        print("xgboost_params", self.params)


    def fit(self, X_train, y_train):
        self.models = []
        for alpha in self.alpha:
            params = self.params['params'].copy()
            params['objective'] = 'reg:quantileerror'
            params['quantile_alpha'] = alpha
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            self.models.append(model)

    def predict(self, X_test):
        predictions = np.array([model.predict(X_test) for model in self.models])
        return predictions.T
    def create_prediction_dataframe(self, predictions, y_test):
        try:
            data = {f'Prediction_{self.alpha[i]}': predictions[:, i] for i in range(len(self.alpha))}
            data['Consumption'] = y_test.values
            return pd.DataFrame(data, index=y_test.index)
        except Exception as e:
            print(f"An error occurred during the creation of the prediction dataframe: {e}")
            raise
