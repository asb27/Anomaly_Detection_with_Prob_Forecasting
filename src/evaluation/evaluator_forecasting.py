import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

class EvaluatorForecasting:
    def __init__(self, config_loader):
        self.config_loader = config_loader

    def evaluate(self, df):

        true_values = df['Consumption']
        predictions = df['Prediction_0.5']

        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predictions)
        mape = mean_absolute_percentage_error(true_values, predictions) * 100

        results = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }

        print(results)

        return results
