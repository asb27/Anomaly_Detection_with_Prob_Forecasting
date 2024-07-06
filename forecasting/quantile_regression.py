import pandas as pd
from sklearn.linear_model import QuantileRegressor
from data_preparation.data_processor import DataProcessor
class QuantileRegression:
    def __init__(self, config_loader, data_processor):
        self.config_loader = config_loader
        self.data_processor = DataProcessor(config_loader)
        self.quantiles = self.config_loader.get_quantiles()
        self.alpha = self.config_loader.get_alpha()
        self.models = {q: QuantileRegressor(quantile=q, alpha=self.config_loader.get_alpha()) for q in self.quantiles}

    def load_and_prepare_data(self, year_train, start_train, end_train, year_test, start_test, end_test):
        print('load dataframe')

        df_train = self.data_processor.create_dataframe(year_train, start_train, end_train,'15min')
        df_test = self.data_processor.create_dataframe(year_test, start_test, end_test,'15min')

        print(df_train.head())
        print(df_test.head())

        X_train = df_train.drop('Consumption', axis=1)
        y_train = df_train['Consumption']
        X_test = df_test.drop('Consumption', axis=1)
        y_test = df_test['Consumption']

        return X_train, y_train, X_test, y_test


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

    def run(self, year_train, start_train, end_train, year_test, start_test, end_test):
        try:
            # Data loading and preparation
            X_train, y_train, X_test, y_test = self.load_and_prepare_data(year_train, start_train, end_train, year_test,
                                                                          start_test, end_test)

            # Model fitting
            self.fit(X_train, y_train)

            # Predictions
            predictions = self.predict(X_test)
            print('Predictions:', predictions)

            # DF creation
            prediction_df = self.create_prediction_dataframe(predictions, y_test)

            return prediction_df
        except Exception as e:
            print(f"An error occurred during the run process: {e}")
            return None