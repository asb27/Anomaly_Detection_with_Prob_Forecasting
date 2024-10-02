from src.data_preparation.data_processor import DataProcessor
from src.factory.model_factory import ModelFactory


class ForecastingBase:
    """"""
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.model_names = config_loader.get_all_model_names()
        self.forecasting_types = config_loader.get_forecasting_types()


    def run(self, model_name, forecasting_type, year_train, start_train, end_train, year_test, start_test, end_test):
        try:
            model_config = self.config_loader.get_forecasting_config(forecasting_type)

            data_processor = DataProcessor(self.config_loader)
            X_train, y_train, X_test, y_test = data_processor.load_and_prepare_data(year_train, start_train, end_train,
                                                                                    year_test, start_test, end_test,model_name)

            model = ModelFactory.get_forecasting_method(forecasting_type, model_config)

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            print(f'Predictions for {model_name}({forecasting_type}):', predictions)

            prediction_df = model.create_prediction_dataframe(predictions, y_test)
            print(prediction_df.head())

            return prediction_df
        except Exception as e:
            print(f"An error occurred during the run process for {model_name}({forecasting_type}): {e}")
            return None

    def make_all_prediction(self, year_train, start_train, end_train, year_test, start_test, end_test):
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

    def run_all_models(self):
        # Retrieve dates from config loader
        training_dates = self.config_loader.get_training_dates()
        testing_dates = self.config_loader.get_testing_dates()

        year_train = training_dates.get('year_train')
        start_train = training_dates.get('start_train')
        end_train = training_dates.get('end_train')

        year_test = testing_dates.get('year_test')
        start_test = testing_dates.get('start_test')
        end_test = testing_dates.get('end_test')

        return self.make_all_prediction(year_train, start_train, end_train, year_test, start_test, end_test)
