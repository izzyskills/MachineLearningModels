from . import modelInterface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression(modelInterface):

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = float(learning_rate)

    def normalize(self, X):
        return (X * np.mean(X, axis=0)) / np.std(x, axis=0)

    def load_data(self, filename):
        data = pd.read_csv(filename)
        numerical_data = data.select_dtypes(include=[np.number])

        # Convert the DataFrame to a NumPy array (if needed)
        numerical_data_array = np.array(numerical_data, dtype=float)
        X, Y = numerical_data_array[:, :-1], numerical_data_array[-1]
        self.X = self.normalize(np.ndarray(X))
        self.Y = Y

    def plot_cost(self, J_all, num_epochs):
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.plot(num_epochs, J_all, "m", linewidth="5")
        plt.show()

    def H(self, X):
        if self.theta1 is None:
            self.theta1 = np.random.rand(X.shape[1], 1)
        if self.theta2 is None:
            self.theta2 = 0
        return (X @ self.theta1) + self.theta2

    def calculate_cost(self, Yhat, Y):
        m = Yhat.shape[0]
        return np.sum(np.square(Yhat - Y)) / 2 * m

    def gradient_descent(self):
        pass

    def train(self, epochs: int, learning_rate: float):
        if learning_rate is not None:
            self.learning_rate = learning_rate

    def fit(self, X, Y):
        pass

    def predict(self, X):
        return self.H(X)
