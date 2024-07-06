import os
import sys
import pandas as pd
import pickle

from config_loader import ConfigLoader
from anomaly_scenarios.anomaly_base import AnomalyBase
from factory.anomaly_factory import AnomalyFactory
from detection_methods.detection_base import DetectionBase
from  evaluation.evaluator import Evaluator

current_dir = os.getcwd()
sys.path.extend([
    os.path.join(current_dir, 'data_preparation'),
    os.path.join(current_dir, 'anomaly_scenarios'),
    os.path.join(current_dir, 'evaluation'),
    os.path.join(current_dir, 'detection_methods'),
    os.path.join(current_dir, 'factory')
])

'''
prediction_df.to_pickle('prepared_data.pkl')
    model.save('quantile_model.pkl')

'''


# Veri ve modeli yükleme
with open(os.path.join(current_dir, 'prepared_data.pkl'), 'rb') as f:
    prediction_df = pickle.load(f)

# Config dosyasını yükleme
config_path = os.path.join(current_dir, 'config.json')
config_loader = ConfigLoader(config_path)

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