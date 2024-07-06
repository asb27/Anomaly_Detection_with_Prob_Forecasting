import json

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        return config

    def get_config(self, section):
        return self.config.get(section, {})

    def get_quantiles(self):
        return self.config['forecasting'].get('quantiles', [0.01, 0.5, 0.99])  # Varsayılan değerler

    def get_alpha(self):
        return self.config['forecasting'].get('alpha', 0.0)
    def get_forecasting_method(self):
        return self.config['forecasting']['type']

    def get_anomaly_types(self):
        return self.config['anomalies'].get('types', [])

    def get_anomaly_periods(self, anomaly_type):
        return self.config['anomalies']['periods'].get(anomaly_type, [])

    def get_detection_parameters(self, method_name):
        return self.config['detection_methods']['parameters'].get(method_name, [])

    def print_detection_methods_and_params(self):
        methods = self.get_config('detection_methods')['methods']
        for method_name in methods:
            params = self.get_detection_parameters(method_name)
            print(f"Method: {method_name}, Parameters: {params}")