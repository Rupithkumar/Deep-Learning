import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Set seed globally
torch.manual_seed(41)

# Convert MNIST images into 4D tensor (number of images, height, width, color channels)
transform_data = transforms.ToTensor()

# Train data
train_data = datasets.MNIST(root='cnn_data', train=True, download=True, transform=transform_data)
# Test data
test_data = datasets.MNIST(root='cnn_data', train=False, download=True, transform=transform_data)

# Create a small batch size for images
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# CNN Model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 240)
        self.fc3 = nn.Linear(240, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Instantiate the model
model = ConvolutionalNetwork()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Training setup
import time
start_time = time.time()
epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# Training loop
for i in range(epochs):
    trn_corr = 0
    for b, (x_train, y_train) in enumerate(train_loader):
        b += 1
        y_pred = model(x_train)  # Get the prediction
        loss = criterion(y_pred, y_train)  # Calculate loss
        predicted = torch.max(y_pred.data, 1)[1]
        batch_correct = (predicted == y_train).sum()
        trn_corr += batch_correct
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if b % 600 == 0:
            print(f'Epoch {i} Batch {b}: Loss: {loss.item()}')

    train_losses.append(loss.item())  # Correctly append loss as float
    train_correct.append(trn_corr)

# Testing loop
with torch.no_grad():
    tes_corr = 0
    for b, (x_test, y_test) in enumerate(test_loader):
        y_val = model(x_test)
        predicted = torch.max(y_val.data, 1)[1]
        tes_corr += (predicted == y_test).sum()
    loss = criterion(y_val, y_test)
    test_losses.append(loss.item())  # Correctly append loss as float
    test_correct.append(tes_corr)

# Calculate total time
current_time = time.time()
total = current_time - start_time
print(f'Training took: {total / 60} minutes')

# Plotting Loss and Accuracy
plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Validation loss")
plt.title("Loss at each epoch")
plt.legend()
plt.show()

plt.plot([t.item() / len(train_data) for t in train_correct], label="Training accuracy")
plt.plot([t.item() / len(test_data) for t in test_correct], label='Validation accuracy')
plt.title("Accuracy at each epoch")
plt.legend()
plt.show()

# Testing with full batch to calculate accuracy
test_load_everything = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for x_test, y_test in test_load_everything:
        y_val = model(x_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

accuracy = correct.item() / len(test_data) * 100
print(f'The correctly predicted among 10000 images is: {correct.item()}')
print(f'The accuracy of the model is: {accuracy}%')

# Taking an image to check
test_image = test_data[4143][0]  # Already a tensor

# Passing the image through the model
model.eval()
with torch.no_grad():
    new_predict = model(test_image.view(1, 1, 28, 28))  # Reshaping the image to fit the model
print(f'The predicted number is: {new_predict.argmax().item()}')
