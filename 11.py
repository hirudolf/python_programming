import torch
import torch.nn as nn  # 딥러닝 구성 하는 모듈
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

'''
ex) x가 몸무게 y가 키
'''

input_size = 1
output_size = 1
num_epochs = 10000
learning_rate = 0.001

x_train = np.array([[60.0], [87.5], [79.0], [101.0], [50.5], [91.0], [77.7]], dtype=np.float32)
y_train = np.array([[165.0], [180.0], [168.0], [180.0], [160.5], [161.0], [171.1]], dtype=np.float32)

x_train = (x_train - np.mean(x_train))/np.std(x_train)
y_train = (y_train - np.mean(y_train))/np.std(y_train)

# LR
model = nn.Linear(input_size, output_size)

# Loss
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 학습
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print('epoch {}/{}, loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label="original data")
plt.plot(x_train, predicted, label="fitted line")
plt.legend()
plt.show()
