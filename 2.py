import numpy as np

inputs = [[1.0,2.0,3.0,2.5,],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8],
          [0.5,3.0,1.0,3.5,],
          [2.0,1.0,2.0,2.5,]
          ]

weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]
           ]
weights2 = [[0.5,0.9,0.5],
           [0.2,-0.71,0.46],
           [-0.28,-0.17,0.67],
           ]
biases = [2.0,3.0,0.5]
biases2 = [1.0,2.0,1.5]

layers_outputs = np.dot(inputs,np.array(weights).T)+biases
print(layers_outputs)
layers_outputs2 = np.dot(layers_outputs,np.array(weights2))+biases2
print(layers_outputs2)

