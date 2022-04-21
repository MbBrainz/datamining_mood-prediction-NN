"""
This file defines a basic neural network and provides a simple train function.

It still needs:
# TODO: Proper quality assessment using test data (Confidence interval etc)


"""

#%%
import os
from statistics import mean
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from MoodDataset import get_dataset_V1, get_dataset_tvt


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_size, nlayers, dropout):
        super(NeuralNetwork, self).__init__()
        
        self.flatten = nn.Flatten()
        layers = []
        # first layer needs in features equal to the input size
        layers.append(nn.Linear(in_features=input_size, out_features=layer_size))
        
        #defining hidden layers 
        for i in range(nlayers):
            layers.append(torch.nn.Linear(layer_size, layer_size))
            layers.append(torch.nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        #defining output layer
        layers.append(nn.Linear(layer_size, 1))
        
        self.linear_relu_stack = nn.Sequential(*layers)
        self.error_list = []


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train_one_epoch(model, training_loader, epoch_index, loss_fn, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (data, targets) in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs = data.view(data.size(0), -1).to(device=DEVICE)
        targets = targets.unsqueeze(1).to(device=DEVICE) 

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return running_loss/(i+1)

# %%

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} DEVICE")

loss_function = nn.MSELoss()
train_loader, val_loader, test_loader = get_dataset_tvt(100, 7)
model = NeuralNetwork(input_size=19, nlayers=4, layer_size=72, dropout=0.42).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.0000128)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))


best_vloss = 1_000_000.
num_epochs = 10000
for epoch in range(num_epochs):
    losses = []
    
    model.train(True)
    avg_loss = train_one_epoch(model, training_loader=train_loader, epoch_index=epoch, loss_fn=loss_function, tb_writer=writer)
    
    model.train(False)
    running_vloss = 0
    
    for i, (vdata, vtargets) in enumerate(val_loader):
        vinputs = vdata.view(vdata.size(0), -1).to(device=DEVICE)
        vtargets = vtargets.unsqueeze(1).to(device=DEVICE) 
        voutputs = model(vinputs)
        vloss = loss_function(voutputs, vtargets)
        running_vloss += vloss
        
    avg_vloss = running_vloss / (i + 1)
    
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    writer.flush()
    
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'trainingdata/model_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)
        
    
#%%
# load best performing model
trainingdata = ["trainingdata/"+ str(x) for x in sorted(os.listdir( 'trainingdata/'))]
saved_model = NeuralNetwork(input_size=19, nlayers=4, layer_size=73, dropout=0.42)
saved_model.load_state_dict(torch.load( trainingdata[-1] ))

#%%

        
#%%
is_gpu = torch.cuda.is_available()
test_loss = 0.0
correct, total = 0,0

for data,label in test_loader:
    if is_gpu:
        data, label = data.cuda(), label.cuda()
    output = model(data)
    for o,l in zip(torch.argmax(output,axis = 1),label):
        if o == l:
            correct += 1
        total += 1
    loss = loss_function(output,label)
    test_loss += loss.item() * data.size(0)
print(f'Testing Loss:{test_loss/len(test_loader)}')
print(f'Correct Predictions: {correct}/{total}')
        
#%%        
# check accuracy
 
def check_accuracy(loader, model: nn.Module):
    model.eval()
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for x, y in loader:
            x= x.to(device=DEVICE)
            y= y.to(device=DEVICE)
            print(f"x shape {x.shape}")
            x = x.reshape(x.shape[0],-1)
            print(f"x shape {x.shape}")
            print(f"y shape {y.shape}")
            print(f"y shape0 {y.shape[0]}")
            
            scores = model(x)
            print(f"scores shape: {scores.shape}")
            error = np.abs(scores.reshape(-1).numpy()-y.numpy())
            num_samples += y.shape[0]
            # print(error.shape)
            # print((error < 0.025))
            num_correct += (error < 0.05).sum()
    

    accuracy = num_correct / num_samples
    print(f"for {num_samples} samples {num_correct} correct, accuracy: {accuracy:.3f}")            
            
            
# %%
check_accuracy(test_loader, model)
# %%
