from . import modelInterface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression(modelInterface.ModelInterface):

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = float(learning_rate)
        self.theta0 = self.theta1 = None

    def normalize(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def load_data(self, filename):
        data = pd.read_csv(filename)
        numerical_data = data.select_dtypes(include=[np.number])

        # Convert the DataFrame to a NumPy array (if needed)
        numerical_data_array = np.array(numerical_data, dtype=float)
        X, Y = numerical_data_array[:, :-1], numerical_data_array[-1]
        self.X = self.normalize(np.array(X))
        self.Y = Y

    def plot_cost(self, J_all, num_epochs):
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.plot(num_epochs, J_all, "m", linewidth="5")
        plt.show()

    def initialize_theta(self, X):
        if self.theta0 is None:
            self.theta0 = 0
        if self.theta1 is None:
            self.theta1 = np.random.rand(X.shape[1], 1)

    def H(self, X):
        self.initialize_theta
        return (X @ self.theta1) + self.theta0

    def calculate_cost(self, Yhat, Y):
        m = Yhat.shape[0]
        return np.sum(np.square((Yhat - Y))) / 2 * m

    def gradient_descent(self):
        m = self.X.shape[0]
        Yhat = self.H(self.X)
        error = Yhat - self.Y
        theta0 = np.sum(error) / m
        theta1 = (self.X.T @ error) / m
        self.theta0 -= self.learning_rate * theta0
        self.theta1 -= self.learning_rate * theta1

    def plot_cost_array(self):
        n_epochs = []
        jplot = []
        count = 0
        for i in self.J_all:
            jplot.append(i)
            n_epochs.append(count)
            count += 1
        jplot = np.array(jplot)
        n_epochs = np.array(n_epochs)
        self.plot_cost(jplot, n_epochs)

    def train(self, epochs: int, learning_rate: float = None):
        self.initialize_theta(self.X)
        self.J_all = []
        if learning_rate is not None:
            self.learning_rate = learning_rate

        for _ in range(0, epochs):
            self.gradient_descent()
            self.J_all.append(self.calculate_cost(self.H(self.X), self.Y))
        self.plot_cost_array()

    def fit(self, X, Y, epochs):
        self.X = self.normalize(X)
        self.Y = np.reshape(Y, (Y.shape[0], 1))

        self.train(epochs)

    def predict(self, X):
        return self.H(X)
