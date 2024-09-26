#importing the modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split

#creating the neural architecture
class Network(nn.Module):
    def __init__(self, seed=42, initial=4, output=3) -> None:
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(initial, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, output)

    #forwarding the message
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  
        return x

#creating the model
model = Network()

#importing the data set
data = pd.read_csv('Iris.csv')

#cleaning the data
data['Species'] = data['Species'].replace('Iris-setosa', 0.0)
data['Species'] = data['Species'].replace('Iris-versicolor', 1.0)
data['Species'] = data['Species'].replace('Iris-virginica', 2.0)

#dividing it into training and testing data
x = data.drop("Species", axis=1)
x = x.drop("Id", axis=1)
y = data['Species']

#converting it into numpy arrays
x = x.values
y = y.values 

#splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

#convert x to float tensors
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)

#convert y to long tensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#criteria to measure the error
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

epochs = 500
losses = []
for i in range(epochs):
    y_pred = model.forward(x_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())  # Corrected this line
    if i % 10 == 0:
        print(f'Epoch: {i} and the loss is: {loss.item()}')  # Use loss.item() for a clean printout
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs),losses)
plt.ylabel("Loss/Error")
plt.xlabel("Epoch")
plt.show()