import pandas as pd

from sklearn.linear_model import QuantileRegressor
class QuantileRegression:
    def __init__(self, model_config):
        self.parameters = model_config
        self.quantiles = self.parameters.get('quantiles', [0.1, 0.5, 0.9])
        self.alpha = self.parameters.get('alpha', 0.0)
        self.models = {q: QuantileRegressor(quantile=q, alpha=self.alpha) for q in self.quantiles}

    def fit(self,X_train, y_train):
        self.models = {
            q: QuantileRegressor(quantile=q, alpha=0).fit(X_train, y_train) for q in self.quantiles
        }

    def predict(self, X_test):
        predictions = {q: self.models[q].predict(X_test) for q in self.quantiles}
        return predictions

    def create_prediction_dataframe(self, predictions, y_test):
        try:
            data = {f'Prediction_{q}': predictions[q] for q in self.quantiles}
            data['Consumption'] = y_test.values
            return pd.DataFrame(data, index=y_test.index)
        except Exception as e:
            print(f"An error occurred during the creation of the prediction dataframe: {e}")
            raise

