import os
import sys
from initial_loader import ConfigLoader
from data_preparation.data_processor import DataProcessor
from factory.model_factory import ModelFactory
from forecasting.forecasting_base import ForecastingBase
from factory.anomaly_factory import AnomalyFactory
from anomaly_scenarios.anomaly_base import AnomalyBase
from detection_methods.detection_base import DetectionBase
from evaluation.evaluator import Evaluator

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'data_preparation'))

# Config dosya yolunu ayarlayın
config_path = os.path.join(current_dir, 'initial.json')
config_loader = ConfigLoader(config_path)

# DataProcessor sınıfını kullanarak veriyi yükleyin ve hazırlayın
data_processor = DataProcessor(config_loader)

year_train = 2019
start_train = '2019-01-01 00:00'
end_train = '2019-12-31 23:59'

year_test = 2020
start_test = '2020-01-01 00:00'
end_test = '2020-12-31 23:59'

# ForecastingBase sınıfını oluşturun
forecasting_base = ForecastingBase(config_loader)

# Tüm modelleri çalıştırarak tahminleri alın
all_predictions = forecasting_base.run_all_models(year_train, start_train, end_train, year_test, start_test, end_test)

for model_type, df in all_predictions.items():
    globals()[f'df_{model_type}'] = df


# Anomali sınıflarını ve senaryolarını yükleyip uygula
anomaly_factory = AnomalyFactory(config_loader)
anomaly_classes = anomaly_factory.get_anomaly_classes()

anomaly_base = AnomalyBase(config_loader)
#all_anomalies_df, prediction_original = anomaly_base.apply_anomalies(prediction_df, anomaly_classes)


detection_base = DetectionBase(config_loader)

# Tüm tespit yöntemlerini uygula ve sonuçları al
#detected_anomalies_df, detection_results = detection_base.apply_detection_methods(all_anomalies_df)

#evaluator = Evaluator(detected_anomalies_df)
#evaluator.evaluate_all_methods()