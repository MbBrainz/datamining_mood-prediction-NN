"""This file contains the temporal neural network defenition

This class still needs:
# TODO: design the network layers of the temporal neural network
# TODO: How to  deal with temporal data
# TODO: Proper quality assessment 

"""

#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from MoodDataset import get_dataset_V1


class TemporalNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO
        
    