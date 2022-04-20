"""
This file defines a basic neural network and provides a simple train function.

It still needs:
# TODO: Proper quality assessment using test data (Confidence interval etc)


"""

#%%
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from MoodDataset import get_dataset_V1

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_size):
        super(NeuralNetwork, self).__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size,layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 1)
            # nn.Softmax(1) #just to get output size 1x1
        )
        # keeping track of error development
        self.error_list = []


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# %%

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} DEVICE")

criterion = nn.MSELoss()
train_loader, test_loader = get_dataset_V1(100, 2)
model = NeuralNetwork(input_size=15, layer_size=50).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 10000
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device=DEVICE)
        targets = targets.unsqueeze(1).to(device=DEVICE) 

        
        # print(f"score predicted: {scores.shape}")
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        # model.error_list.append(loss.detach().numpy())
        
        #bachward
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        print(f"[Epoch{epoch} batch{batch_idx}] loss = {loss}")
        
        
#%%        
# check accuracy
 
# def check_accuracy(loader, model: nn.Module):
#     model.eval()
    
#     with torch.no_grad():
#         for x, y in loader:
#             x= x.to(device=DEVICE)
#             y= y.to(device=DEVICE)
#             x = x.reshape(x.shape[0],-1)
            
#             scores = model(x)
#             _, prediction = scores.max()
            
    
    