import torch
import torch.nn as nn  # short of torch.nn
import numpy as np
import torchvision

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b

y.backward()

print(x.grad)
print(w.grad)
print(b.grad)

x = torch.rand(10, 3)  # (batch_size, input_size)
y = torch.rand(10, 2)  # (batch_size, output_size)

linear = nn.Linear(3, 2)  # Dense(3,2)
print('w:', linear.weight)
print('b:', linear.bias)

criterion = nn.MSELoss()  # Mean Square Error
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

num_epochs = 100
for i in range(num_epochs):
    optimizer.zero_grad()
    pred = linear(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    print(i, loss.item())


x = np.array([[1, 2], [3, 4]])
y = torch.from_numpy(x)
z = y.numpy()

print("A")


train_dataset = torchvision.datasets.CIFAR10(
    root='./data/',
    transform=torchvision.transforms.ToTensor(),
    download=True
)

class Customdataset(torch.utils.data.Dataset):
    def __init__(self):
        #데이터를 초기화, 우리 입력데이터 뭉치가 뭐고 출력(정답지) 데이터 뭉치가 뭐다.

    def __getitem__(self, item):
        #Dataloader Batch단위로 데이터를 뽑음, 100개의 데이터중에 64개를 뽑아야함
    def __len__(self):
        #Dataloader가 접근할수있는 길이

resnet = torchvision.models.resnet18(pretrain=True)
for param in resnet.prarmeters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')


image, label = train_dataset[0]
print(image.size())
print(label)  # image.numpy()[0]

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)
