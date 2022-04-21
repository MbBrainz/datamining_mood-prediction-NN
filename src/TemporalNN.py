"""This file contains the temporal neural network defenition

This class still needs:
# TODO: design the network layers of the temporal neural network
# TODO: How to  deal with temporal data
# TODO: Proper quality assessment 

"""

#%%
from decimal import Decimal
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from MoodDataset import get_dataset_V1, aggr_over_days


class TemporalNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO
    
class GRUModel(nn.Module):
    """GRUModel class extends nn.Module class and works as a constructor for GRUs.

       GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           gru (nn.GRU): The GRU model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of GRUs to our desired output shape.

    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a GRU instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.error_list = []

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
#%%
    
    
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {DEVICE} DEVICE")

# criterion = nn.MSELoss()
# # train_loader, test_loader = get_dataset_timesensitive(100, 1)
# input_dim = 18
# model = GRUModel(input_dim, hidden_dim=10, layer_dim=30, output_dim=1, dropout_prob=0.35).to(DEVICE)
# optimizer = optim.SGD(model.parameters(), lr=1.73E-5)

# train_df=pd.read_csv("../data/train_data_v1.csv", index_col=[0,1])
# train_df["mood(t+1)"] = train_df.groupby(level=0)['mood'].shift(-1).values

# train_df = aggr_over_days(train_df, 1).droplevel(0)
# display(train_df)

# column_list = train_df.columns.tolist()
# column_list.remove("mood")
# column_list.remove("mood(t+1)")
# print(column_list)
# ids = train_df.index.get_level_values(0).drop_duplicates()
# test_ids = ids[23:]
# ids = ids[:23]
# ids

# # %%
# timesteps = 4
# num_epochs = 1000
# for epoch in range(num_epochs):
#     # for batch_idx, (data, targets) in enumerate(train_loader):
#     losses_per_epoch = []
#     for id in ids:
#         print(id)
#         xdata, ydata = get_epoch_data(train_df,column_list,id, timesteps)
        
#         # id_df = train_df.filter(regex=id,axis=0)
#         # xdata = id_df[column_list].values
#         # ydata = id_df["mood(t+1)"]
        
#         data = torch.tensor(xdata, dtype=torch.float).view([ydata.shape[0], -1, input_dim]).to(DEVICE)
#         print(data.shape)
#         targets = torch.tensor(ydata, dtype=torch.float).unsqueeze(1).to(DEVICE)
#         # forward
#         scores = model(data)
#         loss = criterion(scores, targets)
        
#         # model.error_list.append(loss.detach().numpy())
#         losses_per_epoch.append(loss.detach().numpy())
        
#         #bachward
#         optimizer.zero_grad()
#         loss.backward()
        
#         optimizer.step()
#     print(f"[Epoch{epoch} loss = {np.mean(losses_per_epoch):.4f}")
        
#     model.error_list.append(np.mean(losses_per_epoch))
    
# #%%    



# def get_epoch_data(train_df, column_list, timesteps, id):
#     id_df = train_df.filter(regex=id, axis=0)
#     xdata = id_df[column_list].values

#     ydata = id_df["mood(t+1)"].values
#     xdata_temp = []
#     ydata_temp = []
#     for i in range(len(xdata)-timesteps):
#         data_i = xdata[i:i+timesteps]
#         target_i = ydata[i+timesteps]
#         xdata_temp.append(np.array(data_i))
#         ydata_temp.append(target_i)
#     return np.array(xdata_temp), np.array(ydata_temp)

# def get_total_data(train_df, column_list, timesteps, get_epoch_data):
#     ids = train_df.index.get_level_values(0)
#     total_datax = []
#     total_datay = []
#     for id in ids:
#         datax, datay = get_epoch_data(train_df, column_list, timesteps, id)
#         total_datax.append(datax)
#         total_datay.append(datay)

#     total_datax = np.concatenate(total_datax)
#     total_datay = np.concatenate(total_datay)
#     return total_datax, total_datay

# total_datax, total_datay = get_total_data(train_df, column_list, timesteps, get_epoch_data)

# np.shape(total_datax)
# # np.shape(total_datay)

# # get_epoch_data(train_df, column_list, timesteps, id)
# # xdata



        
# #%%
# fig, ax = plt.subplots()
# ax.plot(range(len(model.error_list)), model.error_list)
# ax.set(xlabel="epoch", ylabel="mse")
# # %%
# # randomized data: GRU shoud not perform good here
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {DEVICE} DEVICE")

# criterion = nn.MSELoss()
# train_loader, test_loader = get_dataset_V1(100, 1)
# model = GRUModel(input_dim=15, hidden_dim=10, layer_dim=30, output_dim=1, dropout_prob=0.35).to(DEVICE)
# optimizer = optim.SGD(model.parameters(), lr=1.73E-5)

# num_epochs = 10000
# for epoch in range(num_epochs):
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         # print(data.shape)
#         data = data.view([data.size(0), -1, 15]).to(device=DEVICE)
#         targets = targets.unsqueeze(1).to(device=DEVICE) 

#         # print(f"score predicted: {scores.shape}")
#         # forward
#         scores = model(data)
#         loss = criterion(scores, targets)
#         # model.error_list.append(loss.detach().numpy())
        
#         #bachward
#         optimizer.zero_grad()
#         loss.backward()
        
#         optimizer.step()
#     print(f"[Epoch{epoch} batch{batch_idx}] loss = {loss:.4}")
