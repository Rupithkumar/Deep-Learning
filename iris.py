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
    def __init__(self, seed=35, initial=4, output=3) -> None:
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(initial, 42)
        self.fc2 = nn.Linear(42, 84)
        self.fc3 = nn.Linear(84, 168)
        self.fc4 = nn.Linear(168, 336)
        self.fc5 = nn.Linear(336, 672)
        self.fc6=nn.Linear(672,1344)
        self.fc7=nn.Linear(1344,output)

    #forwarding the message
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        return self.fc7(x)

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=35)

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
    losses.append(loss.detach().numpy())  
    if i % 10 == 0:
        print(f'Epoch: {i} and the loss is: {loss.item()}')  # Use loss.item() for a clean printout
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs),losses)
plt.ylabel("Loss/Error")
plt.xlabel("Epoch")
plt.show()
#tesing the network
# Testing the network
with torch.no_grad(): 
    y_eval = model.forward(x_test)
    loss = criterion(y_eval, y_test)
    print(f'Test Loss: {loss.item()}')

    # Get predictions and calculate accuracy
    _, predicted = torch.max(y_eval, 1)  # Get the predicted class
    accuracy = (predicted == y_test).float().mean()  # Calculate accuracy
    print(f'Test Accuracy: {accuracy.item() * 100:.2f}%')

correct=0
with torch.no_grad():
    for i ,data in enumerate(x_test):
        y_val=model.forward(data)
        if y_test[i]==0:
            x='Iris-setosa'
        elif y_test[i]==1:
            x='Iris-versicolor'
        else:
            x='Iris-virginica'
        
        if y_val.argmax().item()==0:
                y='Iris-setosa'
        elif y_val.argmax().item()==1:
            y='Iris-versicolor'
        else:
            y='Iris-virginica'

        print(f'{i+1}.) {str(y_val)} \t {x} \t {y}')
        if y_val.argmax().item() == y_test[i]:
            correct+=1
print(f'The number of correct answers is: {correct}')
a=float(input("Enter the sepal lenght: "))
b=float(input("Enter the sepal width: "))
c=float(input("Enter the petal lenght: "))
d=float(input("Enter the petal width: "))
iris_data=torch.tensor([a,b,c,d])
with torch.no_grad():
    x_val=model.forward(iris_data)
    if x_val.argmax().item()==0:
            x='Iris-setosa'
    elif x_val.argmax().item()==1:
        x='Iris-versicolor'
    else:
        x='Iris-virginica'
    print(f'The predicted flower is: {x}')

#save the model
torch.save(model.state_dict(),"Iris_prediction_model.pt")
#load the saved model
iris_model=Network()
iris_model.load_state_dict(torch.load("Iris_prediction_model.pt"))
iris_model.eval()