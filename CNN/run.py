from model import GraphConvolution
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device}')

pixel = 648
num_epoch = 5

variable = transforms.Compose(
    [
        transforms.Resize((pixel,pixel)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
        [0.5 for _ in range(3)], [0.2 for _ in range(3)]),
    ]
)
train_data = datasets.ImageFolder(root = 'data/chest_xray/train/', transform =variable)
val_data = datasets.ImageFolder(root = 'data/chest_xray/val/', transform =variable)
test_data = datasets.ImageFolder(root = 'data/chest_xray/test/', transform =variable)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
validloader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True)

model = GraphConvolution(pixel)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(num_epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        if(inputs.size(1)) == 1:
            inputs = torch.squeeze(inputs, 1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 49:
            print(f'[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 50:.3f}')
            running_loss = 0.0

PATH = 'model/trained_cnn_model.pth'
torch.save(model.state_dict(), PATH)
print('Finished Training')

model.to('cpu')
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        # calculate outputs by running images through the network
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the CNN on the test images: {100 * correct // total} %')
