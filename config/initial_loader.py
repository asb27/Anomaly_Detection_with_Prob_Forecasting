import json

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def get_training_dates(self):
        return self.config.get('training_dates', {})

    # Get the testing dates
    def get_testing_dates(self):
        return self.config.get('testing_dates', {})

    # Load the configuration file
    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        return config

    # Get a specific section from the configuration
    def get_config(self, section):
        return self.config.get(section, {})

    def get_all_model_names(self):
        return self.config['data_processor']['model_names']

      # Get the configuration for a specific model
    def get_variables(self, model_name):
        return self.config['data_processor']['models'].get(model_name, {}).get('variables', [])

        # Get the forecasting method types
    def get_forecasting_types(self):
        return self.config['forecasting'].get('type', [])

    # Get the configuration for a specific model
    def get_forecasting_config(self, forecasting_type):
        return self.config['forecasting']['algorithms'].get(forecasting_type, {})

    def get_anomaly_types(self):
        return self.config['anomalies'].get('types', [])

    # Get the periods for a specific anomaly type
    def get_anomaly_periods(self, anomaly_type):
        return self.config['anomalies']['periods'].get(anomaly_type, [])

    # Get the detection parameters for a specific method
    def get_detection_parameters(self, method_name):
        return self.config['detection_methods']['parameters'].get(method_name, [])


