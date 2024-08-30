import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import os, math
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)
class PLData(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.values).to(torch.float32)
        self.y = torch.tensor(y.values).to(torch.long)
        self.y = F.one_hot(self.y, num_classes=3)
        #self.y = torch.reshape(self.y, (len(y.values),))

    def __len__(self):
        if len(self.x) == len(self.y): 
            return len(self.x)
        else:
            raise Exception("x and y do not match")
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


path = "matches.csv"
all_data = pd.read_csv(path, header=0)
x = all_data[["gf","ga","xg","xga","poss","formation","sh","sot","dist","fk","pk","pkatt"]]
x = pd.get_dummies(x, columns = ['formation'])
x = x.fillna(0)
y = all_data[["result"]]
y = y.replace("W", 2)
y = y.replace("L", 0)
y = y.replace("D", 1)
train_x = x[:3900]
train_y = y[:3900]
test_x = x[3900:]
test_y = y[3900:]

train_data = PLData(train_x, train_y)
train_dataloader = DataLoader(dataset=train_data, batch_size=20, shuffle=True)

test_data = PLData(test_x, test_y)
test_dataloader = DataLoader(dataset=test_data, batch_size=20, shuffle=True)

from torch import nn

class PLNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(33, 33),
            nn.ReLU(),
            nn.Linear(33, 33),
            nn.ReLU(),
            nn.Linear(33, 3),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        x = torch.nn.functional.sigmoid(logits)
        return x

model = PLNetwork().to("cpu")
print(model)

learning_rate = 0.05
batch_size = 20
epochs = 5

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        print("\t", pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        current = batch * batch_size + len(X)
        print(loss, current, batch)


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#epochs = 10
#for t in range(epochs):
    #print(f"Epoch {t+1}\n-------------------------------")
    #train_loop(train_dataloader, model, loss_fn, optimizer)
    #test_loop(test_dataloader, model, loss_fn)
#print("Done!")
loss_values = []
for epoch in range(50):
    print("Epoch:", epoch)
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        y = torch.reshape(y, pred.shape).to(torch.float32)
        loss = loss_fn(pred, y)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        print("\tBatch Complete", loss.item())
    print("Epoch Complete")

print("Training Complete")
step = np.linspace(0, 50, 9750)
fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
y_pred = []
y_test = []
total, correct = 0, 0

with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        predicted = np.where(outputs < 0.5, 0, 1)
        y = torch.reshape(y, predicted.shape).to(torch.float32)
        print(predicted)
        print(y.numpy())
        print(predicted.shape, y.shape)
        print(y.size(0))
        print((predicted == y.numpy()))
        input(">")
        y_pred.append(predicted)
        y_test.append(y)
        total += y.size(0)
        correct += (predicted == y.numpy()).sum().item()

print(f'Accuracy of the network on the test instances: {100 * correct // total}%')
