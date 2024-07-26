
class EvaluatorForecasting:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def evaluate(self):
        predictions = self.model.predict(self.data)
        return predictions  # return the predictions

    def evaluate_per_model(self):
        predictions = self.model.predict(self.data)
        return predictions