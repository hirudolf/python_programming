import numpy as np
import random
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method='random'):
        if initialize_method == 'Xavier':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1. / n_inputs)
        elif initialize_method == 'He':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        elif initialize_method == 'Gaussian':
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.layers_outputs = np.dot(inputs, self.weights) + self.biases
        self.layers_outputs = np.maximum(0, self.layers_outputs)
        return self.layers_outputs


X, y = spiral_data(samples=100, classes=3)


p1 = Layer_Dense(2, 3, initialize_method='Xavier')
output = p1.forward(X)
print(output)

p2 = Layer_Dense(3, 5, initialize_method='He')
output2 = p2.forward(output)
print(output2)