from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import pandas as pd


class EvaluatorAD:
    def __init__(self, df, true_anomalies_column='Anomaly'):
        self.df = df
        self.true_anomalies_column = true_anomalies_column

    def evaluate(self, df, detection_column):
        y_true = df[self.true_anomalies_column]
        y_pred = df[detection_column]

        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        #fprr, tprr, _ = roc_curve(y_true, y_pred, zero_division=1)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'auc': auc,


      }

    def evaluate_per_scenario(self, df):
        scenarios = df['Scenario'].unique()
        evaluation_results = {}

        for scenario in scenarios:
            scenario_df = df[df['Scenario'] == scenario]
            detection_columns = [col for col in df.columns if col.startswith('det_')]

            scenario_results = {}
            for column in detection_columns:
                scenario_results[column] = self.evaluate(scenario_df, column)

            evaluation_results[scenario] = scenario_results

        for scenario, results in evaluation_results.items():
            print(f"Results for Scenario- {scenario}:")
            for method, result in results.items():
                print(f"  {method}: {result}")

        return evaluation_results

    def evaluate_per_model(self, df):
        models = {}
        model_results = {}
        detection_columns = [col for col in df.columns if col.startswith('det_')]

        for column in detection_columns:
            model_results[column] = self.evaluate(df, column)


        for method, result in model_results.items():
            print(f"  {method}: {result}")

        return model_results
