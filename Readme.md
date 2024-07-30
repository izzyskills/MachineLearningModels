# Machine Learning Models from Scratch

## Overview

This repository contains implementations of various machine learning models from scratch, without using pre-built libraries such as TensorFlow, PyTorch, or scikit-learn. The goal is to understand the inner workings of these algorithms by implementing them manually. Currently, the repository includes the following models:

- Linear Regression
<!-- - Logistic Regression
- Neural Networks (NN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN) -->

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Models](#models)
   - [Linear Regression](#linear-regression)
   - [Logistic Regression](#logistic-regression)
   - [Neural Networks](#neural-networks)
   - [Convolutional Neural Networks](#convolutional-neural-networks)
   - [Recurrent Neural Networks](#recurrent-neural-networks)
5. [Adding New Models](#adding-new-models)
6. [Contributing](#contributing)
7. [License](#license)

## Project Structure

```
├── models
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── neural_network.py
│   ├── cnn.py
│   ├── rnn.py
│   └── __init__.py
├── utils
│   ├── data_processing.py
│   ├── metrics.py
│   └── __init__.py
├── tests
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   ├── test_neural_network.py
│   ├── test_cnn.py
│   ├── test_rnn.py
│   └── __init__.py
├── README.md
├── requirements.txt
```

<!-- └── setup.py -->

- `models`: Contains the implementations of the models.
- `utils`: Helper functions for data processing, metrics calculation, etc.
- `tests`: Unit tests for the models and utilities.
- `requirements.txt`: List of required Python packages.
<!-- - `setup.py`: Installation script for the project. -->

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To use a specific model, import it from the `models` package and provide the necessary data and parameters. For example, to use Linear Regression:

```python
from models.linear_regression import LinearRegression

# Example usage
model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Models

### Linear Regression

Linear Regression is a simple model for regression tasks. This implementation uses gradient descent for optimization.

### Logistic Regression

Logistic Regression is used for binary classification tasks. It models the probability that an instance belongs to a particular class.

### Neural Networks

A basic implementation of a feedforward neural network. The network's architecture, activation functions, and other parameters can be customized.

### Convolutional Neural Networks

CNNs are particularly effective for image data. This implementation includes convolutional layers, pooling layers, and fully connected layers.

### Recurrent Neural Networks

RNNs are designed for sequential data. This implementation includes simple RNN layers and can be extended to include more advanced architectures like LSTM or GRU.

## Adding New Models

To add a new model, follow these steps:

1. Create a new file in the `models` directory, e.g., `my_new_model.py`.
2. Implement the model class, ensuring it has `fit`, `predict`, and other necessary methods.
3. Add unit tests for your model in the `tests` directory.
4. Update the documentation and examples accordingly.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes. Make sure to follow the existing code style and include tests for new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
