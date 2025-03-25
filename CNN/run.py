from model import GraphConvolution
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device}')

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