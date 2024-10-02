import os
import sys
import pandas as pd
import tables
import warnings
import pickle

from config.initial_loader import ConfigLoader
from src.data_preparation.data_processor import DataProcessor
from src.forecasting.forecasting_base import ForecastingBase
from src.factory.anomaly_factory import AnomalyFactory
from src.anomaly_scenarios.anomaly_base import AnomalyBase
from src.detection_methods.det_base import DetectionBase
from src.evaluation.evaluator_base import EvaluatorBase

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'data_preparation'))

config_path = os.path.join(current_dir, 'config\\initial.json')
config_loader = ConfigLoader(config_path)

data_processor = DataProcessor(config_loader)
forecasting_base = ForecastingBase(config_loader)

# all_predictions load
with open('all_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# all_predictions = forecasting_base.run_all_models()

# for model_type, df in all_predictions.items():
#   globals()[f'df_{model_type}'] = df


anomaly_factory = AnomalyFactory(config_loader)
anomaly_classes = anomaly_factory.get_anomaly_classes()
anomaly_base = AnomalyBase(config_loader)

def remove_rows_in_date_range(dfs_dict: dict, start_date: str, end_date: str) -> dict:
    for key, df in dfs_dict.items():
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError(f"DataFrame {key} index must be of DateTime type.")
        # Filter out rows within the specified date range
        dfs_dict[key] = df.loc[~((df.index >= start_date) & (df.index <= end_date))]
    return dfs_dict


# 08-2020   08-12 Anomaly
#all_predictions = remove_rows_in_date_range(all_predictions, "2020-08-09", "2020-08-12")

#all_predictions = remove_rows_in_date_range(all_predictions, "2020-12-20", "2020-12-26")

'''
for key, (filtered_df, df_original) in results.items():
    print(f"Results for {key}:")
    print("Filtered DataFrame:")
    print(filtered_df.head())
    print("Original DataFrame:")
    print(df_original.head())
'''

results = anomaly_base.apply_anomalies(all_predictions, anomaly_classes)

dict_with_anomaly = {}

for key, value in results.items():
    if isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], pd.DataFrame):
        dict_with_anomaly[key] = value[0]


detection_base = DetectionBase(config_loader)
detection_results = detection_base.all_detection_methods(dict_with_anomaly)

evaluator_base = EvaluatorBase(config_loader)
evaluation_results = evaluator_base.evaluate_all(detection_results)

for model_type, df in detection_results.items():
    globals()[f'df_{model_type}'] = df

'''
import pickle

# all_predictions save
with open('all_predictions.pkl', 'wb') as f:
    pickle.dump(all_predictions, f)


import pickle

# all_predictions load
with open('all_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)


'''

