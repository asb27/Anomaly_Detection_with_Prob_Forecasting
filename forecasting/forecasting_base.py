from initial_loader import ConfigLoader
from data_preparation.data_processor import DataProcessor
from factory.model_factory import ModelFactory
import pandas as pd
import os

class ForecastingBase:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.model_names = config_loader.get_all_model_names()
        self.forecasting_types = config_loader.get_forecasting_types()


    def run(self, model_name, forecasting_type, year_train, start_train, end_train, year_test, start_test, end_test):
        try:
            # Get model configuration
            model_config = self.config_loader.get_forecasting_config(forecasting_type)

            # Data loading and preparation
            data_processor = DataProcessor(self.config_loader)
            X_train, y_train, X_test, y_test = data_processor.load_and_prepare_data(year_train, start_train, end_train,
                                                                                    year_test, start_test, end_test,model_name)

            # Model creation and fitting
            model = ModelFactory.get_forecasting_method(forecasting_type, model_config)

            model.fit(X_train, y_train)

            # Predictions
            predictions = model.predict(X_test)
            print(f'Predictions for {model_name}({forecasting_type}):', predictions)

            # Create prediction DataFrame
            prediction_df = model.create_prediction_dataframe(predictions, y_test)
            print(prediction_df.head())

            return prediction_df
        except Exception as e:
            print(f"An error occurred during the run process for {model_name}({forecasting_type}): {e}")
            return None

    def run_all_models(self, year_train, start_train, end_train, year_test, start_test, end_test):
        all_predictions = {}

        for model_name in self.model_names:
            for forecasting_type in self.forecasting_types:
                try:
                    prediction_df = self.run(model_name, forecasting_type, year_train, start_train, end_train,
                                             year_test, start_test, end_test)
                    if prediction_df is not None:
                        print(f"Prediction results for {model_name}({forecasting_type}):")
                        print(prediction_df.head())

                        # add the prediction results to the dictionary
                        all_predictions[f"{model_name}_{forecasting_type}"] = prediction_df

                except Exception as e:
                    print(f"An error occurred during the run process for {model_name}({forecasting_type}): {e}")

        return all_predictions

