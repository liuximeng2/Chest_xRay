from model import GraphConvolution
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
<<<<<<< HEAD
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score
=======
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
>>>>>>> 438e090b02d0babb89fcb774ed9a1ae6fda363e9

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device}')

<<<<<<< HEAD
pixel = 64
num_epoch = 20

# read data from .npy file: Data/Data Transformed/X_train.npy
x_train = np.load('Data/Data Transformed/X_train.npy')
y_train = np.load('Data/Data Transformed/y_train.npy')
x_test = np.load('Data/Data Transformed/X_test.npy')
y_test = np.load('Data/Data Transformed/y_test.npy')
# load the data into the DataLoader
train_data = TensorDataset(torch.Tensor(x_train).permute(0, 3, 1, 2), torch.LongTensor(y_train))
test_data = TensorDataset(torch.Tensor(x_test).permute(0, 3, 1, 2), torch.LongTensor(y_test))
trainloader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=4, shuffle=False)

print('Data loaded, Trainning set size is', len(train_data), 'Test set size is', len(test_data))

model = GraphConvolution(pixel, kernel_size = 3)
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Lists to store metrics history
train_losses = []
train_f1_scores = []
test_f1_scores = []

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.float()

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        running_loss += loss.item()
    
    # Calculate epoch metrics
    model.eval()
    epoch_loss = 0.0
    all_train_preds = []
    all_train_labels = []
    all_test_preds = []
    all_test_labels = []
    
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            if(inputs.size(1)) == 1:
                inputs = torch.squeeze(inputs, 1)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            predicted = torch.round(outputs)
            
            # Collect predictions and labels for F1 calculation
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            predicted = torch.round(outputs)
            
            # Collect predictions and labels for F1 calculation
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
    
    train_loss = epoch_loss / len(trainloader)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    
    train_losses.append(train_loss)
    train_f1_scores.append(train_f1)
    test_f1_scores.append(test_f1)
    
    print(f'Epoch {epoch+1}/{num_epoch} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}')

print('Finished Training')
=======
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
>>>>>>> 438e090b02d0babb89fcb774ed9a1ae6fda363e9
