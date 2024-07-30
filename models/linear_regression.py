from . import modelInterface
import numpy as np


class LinearRegression(modelInterface):

    def __init__(self, learning_rate: float):
        self.learning_rate = float(learning_rate)

    def normalize(self):
        pass

    def cost_function(self, Yhat, Y):
        pass

    def gradient_descent(self):
        pass

    def train(self):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, X, Y):
        pass
