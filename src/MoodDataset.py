#%%
from math import ceil, floor
from sqlalchemy import false
import torch
from torch import nn, optim, split
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#TODO: The dataset is now features and targets of the same dat in the same row
# This should be changed to one of these:
# - Sum 5 consequtive days and ad the mood of the rest
# - put the mood of the next day on the previous row ( shape will be N-1 )
class MoodDatasetV1(Dataset):
    
    def __init__(self,file_name , day_window=1, timesteps=1, train=True,):
        self.train_df=pd.read_csv(file_name, index_col=[0,1], parse_dates=True)
        target = "mood(t+1)"
        # display(self.train_df)

        # shifts the mood of t+1 to the mood of t
        self.train_df["mood(t+1)"] = self.train_df.groupby(level=0)['mood'].shift(-1).values
        column_list = self.train_df.columns.tolist()
        column_list.remove(target)
        
        x=self.train_df[column_list].values
        y=self.train_df[target].values
        
        if day_window > 1 and timesteps > 1:
            raise ValueError("Day window and timesteps can both be larger then 1")
        
        # performs window rolling to get data in form t(1->6), t(2-7) etc.       
        elif day_window > 1:
            self.train_df = aggr_over_days(self.train_df, day_window).reset_index(drop=True)
            x=self.train_df[column_list].values
            y=self.train_df[target].values
            
        
        elif timesteps > 1:
            column_list.remove("mood") # for GRU we dont want the mood to be a variable, because its in the memory
            xdata, ydata = get_timestep_data(self.train_df,column_list=column_list, timesteps=timesteps)
            
            x = xdata
            y = ydata
        
        self.input_size = len(column_list)

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)
        
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

    
def _get_timestep_data_per_id(train_df, column_list, timesteps, id):
    id_df = train_df.filter(regex=id,axis=0)
    xdata = id_df[column_list].values

    ydata = id_df["mood(t+1)"].values
    xdata_temp = []
    ydata_temp = []
    for i in range(len(xdata)-timesteps):
        data_i = xdata[i:i+timesteps]
        target_i = ydata[i+timesteps]
        xdata_temp.append(np.array(data_i))
        ydata_temp.append(target_i)
    return np.array(xdata_temp), np.array(ydata_temp)

def get_timestep_data(train_df, column_list, timesteps, ):
    ids = train_df.index.get_level_values(0)

    total_datax = []
    total_datay = []
    for id in ids:
        datax, datay = _get_timestep_data_per_id(train_df, column_list, timesteps, id)
        total_datax.append(datax)
        total_datay.append(datay)

    total_datax = np.concatenate(total_datax)
    total_datay = np.concatenate(total_datay)
    return total_datax, total_datay

    
def aggr_over_days(train_df, n_days=1) -> pd.DataFrame:
    moods = train_df["mood"].values
    windowed_df = train_df.groupby(level=0).rolling(n_days).mean()
    windowed_df["mood"] = moods
    windowed_df = windowed_df.dropna()
    return windowed_df

def get_dataset_V1(batch_size, day_window=1, timesteps=1):
    dataset = MoodDatasetV1("../data/train_data_v1.csv", day_window, timesteps=timesteps)
    # trainset, testset = split(dataset, [ceil(0.7*len(dataset)),floor(0.3*len(dataset))])
    trainset, testset = random_split(dataset, lengths=[ceil(0.7*len(dataset)),floor(0.3*len(dataset))], generator=torch.Generator().manual_seed(155))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def get_dataset_tvt(batch_size, day_window=1, timesteps=1):
    """returns train_loader(70%), validation_loader(10%) and test_loader(20%)

    Args:
        batch_size (_type_): _description_
        day_window (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    dataset = MoodDatasetV1("../data/train_data_v1.csv", day_window, timesteps)
    datasize = len(dataset)

    # checks if split numbers are correct
    train_size, val_size, test_size = ceil(0.7*datasize),ceil(0.1*datasize),floor(0.2*datasize)
    while (train_size + val_size + test_size) % datasize > 0:
        train_size -= 1
    
    # random splits the data set into train, val and test
    trainset, val_set, testset = random_split(dataset, lengths=[train_size, val_size, test_size], generator=torch.Generator().manual_seed(155))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader


dataset = MoodDatasetV1("../data/train_data_v1.csv", day_window=1)
dataset = MoodDatasetV1("../data/train_data_v1.csv", day_window=2)
dataset = MoodDatasetV1("../data/train_data_v1.csv", timesteps=1)
dataset = MoodDatasetV1("../data/train_data_v1.csv", timesteps=1)

#%%
get_dataset_tvt(100, 1, 2)

# loaders = get_dataset_V1(100,1)


# %%

# %%
