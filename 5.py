import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Dense Layer Class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.5  # 초기 가중치
        self.biases = np.random.randn(1, n_neurons) * 0.5  # 초기 바이어스

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs


# Activation ReLu Class
class Activation_Relu:
    def forward(self, inputs):
        return np.maximum(0, inputs)


# Activation Softmax Class
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities


## Loss_CategoricalCrossentropy
class Cross_entroy:
    def forward(predictions, targets):
        '''
        :param predictions: dense layer output => softmax 취한 출력
        :param targets: 정답지 one-hot encording
        :return: categorical cross entropy loss 연산값
        '''
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        ## clip : 범위를 정해줌 // e = 10, e-7 = 10^-7
        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)), targets]
        else:
            correct_confidences = np.sum(predictions * targets, axis=1)

        negative_log_likelhoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelhoods)


# Create dataset
X, y = spiral_data(samples=100, classes=3)

## forward

layer = Layer_Dense(2, 3)

activation_relu = Activation_Relu()
activation_softmax = Activation_Softmax()

layer.forward(X)
activation_relu.forward(layer.output)
activation_softmax.forward(activation_relu.output)

# loss calculation
loss = Layer_Dense.Cross_entropy(activation_softmax.output, y)

# print loss
print("Categorical Cross-Entropy Loss:", loss)
