from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd


class Evaluator:
    def __init__(self, df, true_anomalies_column='Anomaly'):
        self.df = df
        self.true_anomalies_column = true_anomalies_column

    def evaluate(self,df, detection_column):
        y_true = df[self.true_anomalies_column]
        y_pred = df[detection_column]

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        }

    def visualize(self, detection_columns):
        plt.figure(figsize=(12, 6))

        # Consumption ve Anomaly_Consumption plot
        plt.plot(self.df.index, self.df['Consumption'], 'b-', label='Consumption')
        plt.plot(self.df.index, self.df['Anomaly_Consumption'], 'r-', label='Anomaly Consumption')

        # Quantile bands plot
        quantile_columns = [col for col in self.df.columns if col.startswith('Prediction_')]
        for col in quantile_columns:
            plt.plot(self.df.index, self.df[col], 'k-', alpha=0.3)

        # Anomaly detections plot
        for index, row in self.df.iterrows():
            if row[self.true_anomalies_column] == 1:
                detection = row[detection_columns].sum()
                if detection == len(detection_columns):
                    plt.plot(index, row['Anomaly_Consumption'], 'go')  # green if all methods detected
                elif detection == 0:
                    plt.plot(index, row['Anomaly_Consumption'], 'ro')  # red if none detected
                elif row[detection_columns[0]] == 1 and row[detection_columns[1]] == 0:
                    plt.plot(index, row['Anomaly_Consumption'], 'yo')  # yellow if only method 1 detected
                elif row[detection_columns[0]] == 0 and row[detection_columns[1]] == 1:
                    plt.plot(index, row['Anomaly_Consumption'], 'co')  # cyan if only method 2 detected

        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Anomaly Detection Results')
        plt.show()

    def evaluate_per_scenario(self):
        scenarios = self.df['Scenario'].unique()
        evaluation_results = {}

        for scenario in scenarios:
            scenario_df = self.df[self.df['Scenario'] == scenario]
            detection_columns = [col for col in self.df.columns if col.startswith('Detect_')]

            scenario_results = {}
            for column in detection_columns:
                scenario_results[column] = self.evaluate(scenario_df, column)

            evaluation_results[scenario] = scenario_results

        for scenario, results in evaluation_results.items():
            print(f"Results for {scenario}:")
            for method, result in results.items():
                print(f"  {method}: {result}")

