#%%
from math import ceil, floor
from sqlalchemy import false
import torch
from torch import nn, optim, split
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import pandas as pd

#TODO: The dataset is now features and targets of the same dat in the same row
# This should be changed to one of these:
# - Sum 5 consequtive days and ad the mood of the rest
# - put the mood of the next day on the previous row ( shape will be N-1 )
class MoodDatasetV1(Dataset):
    
    def __init__(self,file_name , day_window, train=True,):
        self.train_df=pd.read_csv(file_name, index_col=[0,1], parse_dates=True)
        target = "mood(t+1)"
        # display(self.train_df)

        # shifts the mood of t+1 to the mood of t
        self.train_df["mood(t+1)"] = self.train_df.groupby(level=0)['mood'].shift(-1).values

        # performs window rolling to get data in form t(1->6), t(2-7) etc.       
        self.train_df = aggr_over_days(self.train_df, day_window).reset_index(drop=True)
        # self.train_df = self.aggr_over_days(day_window).droplevel([0,1]).reset_index(drop=False)
        # print(self.train_df.dtypes)
        # self.train_df["date"] = self.train_df["date"].astype('datetime64[s]').astype('int')
        
        # print(self.train_df)
        # self.train_df = self.train_df.index
        column_list = self.train_df.columns.tolist()

        column_list.remove(target)

        x=self.train_df[column_list].values
        y=self.train_df[target].values
        
        self.input_size = len(column_list)

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)
        
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]
    
def aggr_over_days(train_df, n_days=1) -> pd.DataFrame:
    moods = train_df["mood"].values
    windowed_df = train_df.groupby(level=0).rolling(n_days).mean()
    windowed_df["mood"] = moods
    windowed_df = windowed_df.dropna()
    return windowed_df

def get_dataset_V1(batch_size, day_window=1):
    dataset = MoodDatasetV1("../data/train_data_v1.csv", day_window)
    # trainset, testset = split(dataset, [ceil(0.7*len(dataset)),floor(0.3*len(dataset))])
    trainset, testset = random_split(dataset, lengths=[ceil(0.7*len(dataset)),floor(0.3*len(dataset))], generator=torch.Generator().manual_seed(155))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# dataset = MoodDatasetV1("../data/train_data_v1.csv", 1)

# loaders = get_dataset_V1(100,1)


# %%

# %%
