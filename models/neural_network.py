from . import modelInterface
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(modelInterface.ModelInterface):
    def __init__(self, learning_rate:float = 0.1):
        self.learning_rate = float(learning_rate)
        self.theta0 = self.theta1 = None

    def normalize(self,X):
        return (X -np.mean(X,axis=0))/np.std(X,axis=0)

    def relu(Self,X):
        return max(0,X)

    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
