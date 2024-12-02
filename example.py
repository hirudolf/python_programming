import random

def init_weights(inputs):
    weights = list()
    for i in range(len(inputs)):
        weights.append(random.uniform(-1, 1))
    return weights

def cal(inputs, weights, bias):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i]*weights[i]
    output = output + bias
    return output

def cal_neuron(num_neuron, inputs):
    outputs = list()
    for i in range(num_neuron):
        weights = init_weights(inputs)
        output = cal(inputs, weights, bias)
        outputs.append(output)
    return outputs

inputs = [1.0,2.0,3.0]
num_neuron=5
bias = 5
outputs = cal_neuron(num_neuron, inputs)
print(outputs)