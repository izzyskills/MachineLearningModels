from platform import node
from . import modelInterface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression(modelInterface.ModelInterface):
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.theta0 = self.theta1 = None

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def gradient_descent(self):
        pass

    def fit(self, X, Y):
        pass

    def calculate_cost(self, Yhat, Y):
        return -np.sum(Y * np.log(Yhat) + (Y - Yhat) * np.log(1 - Yhat))

    def predict(self):
        pass

    def train(self):
        pass

    def load_data(self):
        pass

    def plot_cost(self):
        pass
