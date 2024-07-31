import unittest
import numpy as np
from models.linear_regression import LinearRegression

# Replace 'your_module' with the actual module name


class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        # Setup code to run before each test
        self.model = LinearRegression(learning_rate=0.1)
        # Example dataset for testing
        self.X_train = np.array([[1, 2], [3, 4], [5, 6]])
        self.Y_train = np.array([3, 7, 11])
        self.X_test = np.array([[7, 8], [9, 10]])
        self.Y_test = np.array([15, 19])
        self.model.fit(self.X_train, self.Y_train, epochs=1000)

    def test_initialize_theta(self):
        self.model.initialize_theta(self.X_train)
        self.assertIsNotNone(self.model.theta0, "Theta0 should not be None")
        self.assertIsNotNone(self.model.theta1, "Theta1 should not be None")

    def test_normalize(self):
        normalized_X = self.model.normalize(self.X_train)
        self.assertTrue(
            np.allclose(np.mean(normalized_X, axis=0), 0), "Mean should be close to 0"
        )
        self.assertTrue(
            np.allclose(np.std(normalized_X, axis=0), 1), "Std should be close to 1"
        )

    def test_H(self):
        self.model.initialize_theta(self.X_train)
        predictions = self.model.H(self.X_train)
        self.assertEqual(
            predictions.shape, (self.X_train.shape[0],), "Predictions shape mismatch"
        )

    def test_calculate_cost(self):
        Yhat = self.model.H(self.X_train)
        cost = self.model.calculate_cost(Yhat, self.Y_train)
        self.assertGreater(cost, 0, "Cost should be greater than 0")

    def test_gradient_descent(self):
        old_theta0 = self.model.theta0
        old_theta1 = self.model.theta1
        self.model.gradient_descent()
        self.assertNotEqual(
            self.model.theta0, old_theta0, "Theta0 should have been updated"
        )
        self.assertNotEqual(
            self.model.theta1, old_theta1, "Theta1 should have been updated"
        )

    def test_train(self):
        old_cost = self.model.calculate_cost(self.model.H(self.X_train), self.Y_train)
        self.model.train(epochs=100, learning_rate=0.1)
        new_cost = self.model.calculate_cost(self.model.H(self.X_train), self.Y_train)
        self.assertLess(new_cost, old_cost, "Cost should decrease after training")

    def test_predict(self):
        predictions = self.model.predict(self.X_test)
        self.assertEqual(
            predictions.shape, self.Y_test.shape, "Predictions shape mismatch"
        )

    def test_fit(self):
        self.model.fit(self.X_train, self.Y_train, epochs=1000)
        self.assertIsNotNone(self.model.X, "X should not be None after fitting")
        self.assertIsNotNone(self.model.Y, "Y should not be None after fitting")


if __name__ == "__main__":
    unittest.main()
