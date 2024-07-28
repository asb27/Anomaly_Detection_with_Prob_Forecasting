import os
import sys
import pandas as pd
import tables
import warnings

from config.initial_loader import ConfigLoader
from src.data_preparation.data_processor import DataProcessor
from src.forecasting.forecasting_base import ForecastingBase
from src.factory.anomaly_factory import AnomalyFactory
from src.anomaly_scenarios.anomaly_base import AnomalyBase
from src.detection_methods.detection_base import DetectionBase
from src.evaluation.evaluator_base import EvaluatorBase

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'data_preparation'))

config_path = os.path.join(current_dir, 'config\\initial.json')
config_loader = ConfigLoader(config_path)

data_processor = DataProcessor(config_loader)
forecasting_base = ForecastingBase(config_loader)

all_predictions = forecasting_base.run_all_models()

#for model_type, df in all_predictions.items():
#   globals()[f'df_{model_type}'] = df


anomaly_factory = AnomalyFactory(config_loader)
anomaly_classes = anomaly_factory.get_anomaly_classes()
anomaly_base = AnomalyBase(config_loader)

results = anomaly_base.apply_anomalies(all_predictions, anomaly_classes)

'''
for key, (filtered_df, df_original) in results.items():
    print(f"Results for {key}:")
    print("Filtered DataFrame:")
    print(filtered_df.head())
    print("Original DataFrame:")
    print(df_original.head())
'''

dict_with_anomaly = {}

for key, value in results.items():
    if isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], pd.DataFrame):
        dict_with_anomaly[key] = value[0]


detection_base = DetectionBase(config_loader)
detection_results = detection_base.all_detection_methods(dict_with_anomaly)

evaluator_base = EvaluatorBase(config_loader)
evaluation_results = evaluator_base.evaluate_all(detection_results)

