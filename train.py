import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from base import convertToDataset
from base import NeuralNetwork

training = pd.read_csv("train.csv")
testing = pd.read_csv("test.csv")

train_dataloader = DataLoader(convertToDataset(training), batch_size=64, shuffle=True)
test_dataloader = DataLoader(convertToDataset(testing), batch_size=64, shuffle=True)

batch_size = 64

device = (    
    "cuda"
   if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

best_acc = -np.inf

# 70 input -> 30 hidden layer
model = NeuralNetwork(70, 30).to(device)

# Binary classification calls for BCE criterion
criterion = torch.nn.BCELoss()

# High learning rate tuning for quick model creation
optimizer = torch.optim.SGD(model.parameters(), lr = 0.9, momentum=0.2)


def train(dataloader, model, loss_fn, optimizer):
    size = len(training)
    
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Test our model and calculate correctness rate
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.squeeze(), y).item()
            # print(f"{numpy.round(pred.squeeze()[0].item())} {y[0]}")
            correct += (pred.squeeze() - y).mean().sqrt()

    test_loss /= num_batches
    correct /= size
    correct = 1 - correct
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")        

epochs = 5000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")