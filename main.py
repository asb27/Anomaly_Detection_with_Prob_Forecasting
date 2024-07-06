import os
import sys

from config_loader import ConfigLoader
from data_preparation.data_processor import DataProcessor
from factory.model_factory import ModelFactory


from factory.anomaly_factory import AnomalyFactory
from anomaly_scenarios.anomaly_base import AnomalyBase
from detection_methods.detection_base import DetectionBase
from evaluation.evaluator import Evaluator

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'data_preparation'))

# Config dosya yolunu ayarlayın
config_path = os.path.join(current_dir, 'config.json')
config_loader = ConfigLoader(config_path)

# DataProcessor sınıfını kullanarak veriyi yükleyin ve hazırlayın
data_processor = DataProcessor(config_loader)

year_train = 2019
start_train = '2019-01-01 00:00'
end_train = '2019-12-31 23:59'

year_test = 2020
start_test = '2020-01-01 00:00'
end_test = '2020-12-31 23:59'

model = ModelFactory.get_forecasting_method(config_loader)
prediction_df  = model.run(year_train, start_train, end_train, year_test, start_test, end_test)

# Anomali sınıflarını ve senaryolarını yükleyip uygula
anomaly_factory = AnomalyFactory(config_loader)
anomaly_classes = anomaly_factory.get_anomaly_classes()

anomaly_base = AnomalyBase(config_loader)
all_anomalies_df, prediction_original = anomaly_base.apply_anomalies(prediction_df, anomaly_classes)


detection_base = DetectionBase(config_loader)

# Tüm tespit yöntemlerini uygula ve sonuçları al
detected_anomalies_df, detection_results = detection_base.apply_detection_methods(all_anomalies_df)

evaluator = Evaluator(detected_anomalies_df)
evaluator.evaluate_all_methods()