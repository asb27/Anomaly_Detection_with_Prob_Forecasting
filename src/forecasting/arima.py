import pandas as pd
import xgboost as xgb
import numpy as np

class Arima:
    def __init__(self, config_loader):
        self.params = config_loader
        print("arima_params", self.params)
        self.model = None
