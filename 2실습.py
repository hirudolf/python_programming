import numpy as np
import random
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.biases = np.random.uniform(0, 1, (1, n_neurons))

    def forward(self, inputs):
        self.layers_outputs = np.dot(inputs, self.weights) + self.biases
        return self.layers_outputs


X, y = spiral_data(samples=100, classes=3)


p1 = Layer_Dense(2, 3)
output = p1.forward(X)
print(output)

p2 = Layer_Dense(3, 5)
output2 = p2.forward(output)
print(output2)