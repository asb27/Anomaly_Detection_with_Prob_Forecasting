from src.evaluation.evaluator_forecasting import EvaluatorForecasting
from src.evaluation.evaluator_AD import EvaluatorAD
from src.evaluation.visualizer import Visualizer

class EvaluatorBase:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.forecasting_evaluator = EvaluatorForecasting(config_loader)
        self.anomaly_evaluator = EvaluatorAD(config_loader)
        self.visualizer= Visualizer(config_loader)

    def evaluate_all(self, all_dfs):
        if not isinstance(all_dfs, dict):
            raise ValueError("Input should be a dictionary of DataFrames.")

        evaluation_results = {}

        for key, df in all_dfs.items():
            print(f"\n\nAnomaly Evaluating results for {key}:")

            anomaly_eval_result = self.anomaly_evaluator.evaluate_per_model(df)

            print(f"Forecasting Evaluating results for {key}:")

            forecasting_eval_result = self.forecasting_evaluator.evaluate(df)

            evaluation_results[key] = {
                'anomaly_evaluation': anomaly_eval_result,
                'forecasting_evaluation': forecasting_eval_result
            }

        return evaluation_results

