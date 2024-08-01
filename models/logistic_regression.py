from platform import node
from . import modelInterface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression(modelInterface.ModelInterface):
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.theta0 = self.theta1 = None

    def normalize(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def load_data(self, filename):
        data = pd.read_csv(filename)
        numerical_data = data.select_dtypes(include=[np.number])

        # Convert the DataFrame to a NumPy array (if needed)
        numerical_data_array = np.array(numerical_data, dtype=float)
        X, Y = self.normalize(numerical_data_array[:, :-1]), numerical_data_array[-1]
        return X, Y

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def initialize_weight(self, X):
        self.theta0 = np.random.rand(X.shape[1], 1)
        self.theta1 = 0

    def gradient_descent(self, X, Y):
        m = X.shape[0]
        temp = self.predict(X) - Y
        theta1 = (X.T @ temp) / m
        theta0 = np.sum(temp) / m
        self.theta1 -= self.learning_rate * theta1
        self.theta0 -= self.learning_rate * theta0

    def fit(self, X, Y, epochs: int, learning_rate: float = None):
        self.initialize_weight(X)
        j_cost = np.array(dtype=float)
        count = np.array(dtype=int)
        for i in range(epochs):
            self.gradient_descent(X, Y)
            j_cost.append(self.calculate_cost(self.predict(X), Y))
            count.append(i)
        self.plot_cost(j_cost, count)

    def plot_cost(self, J_all, num_epochs):
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.plot(num_epochs, J_all, "m", linewidth="5")
        plt.show()

    def calculate_cost(self, Yhat, Y):
        return -np.sum(Y * np.log(Yhat) + (Y - Yhat) * np.log(1 - Yhat))

    def predict(self, X):
        return self.sigmoid(X @ self.theta1 + self.theta1)

    def load_data(self):
        pass
