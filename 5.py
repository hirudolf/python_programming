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
        self.output = np.maximum(0, inputs)


# Activation Softmax Class
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities


# Loss_CategoricalCrossentropy Class
class Cross_entropy:
    def forward(self, predictions, targets):
        '''
        :param predictions: dense layer output => softmax 취한 출력
        :param targets: 정답지 one-hot encoding 또는 인덱스
        :return: categorical cross entropy loss 연산값
        '''
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)), targets]
        else:
            correct_confidences = np.sum(predictions * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)


# 데이터셋 생성
X, y = spiral_data(samples=100, classes=3)

## forward
dense1 = Layer_Dense(2, 8)
dense2 = Layer_Dense(8, 8)
dense3 = Layer_Dense(8, 3)

activation1 = Activation_Relu()
activation2 = Activation_Relu()
activation_softmax = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.outputs)
dense2.forward(activation1.output)
activation2.forward(dense2.outputs)
dense3.forward(activation2.output)
activation_softmax.forward(dense3.outputs)

# loss calculation
loss_function = Cross_entropy()
loss = loss_function.forward(activation_softmax.output, y)

# print loss
print("Categorical Cross-Entropy Loss:", loss)
