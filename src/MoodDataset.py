#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd

#TODO: The dataset is now features and targets of the same dat in the same row
# This should be changed to one of these:
# - Sum 5 consequtive days and ad the mood of the rest
# - put the mood of the next day on the previous row ( shape will be N-1 )
class MoodDatasetV1(Dataset):
    
    def __init__(self,file_name , day_window, train=True,):
        self.train_df=pd.read_csv(file_name, index_col=[0,1])
        target = "mood"
        # display(self.train_df)

        # shifts the mood of t+1 to the mood of t
        self.train_df["mood"] = self.train_df.groupby(level=0)['mood'].shift(-1).values

        
        self.train_df = self.aggr_over_days(day_window).reset_index(drop=True)
        # display(self.train_df.index)
        
        column_list = self.train_df.columns.tolist()
        column_list.remove(target)
        # print(self.train_df[column_list].values)

        x=self.train_df[column_list].values
        y=self.train_df[target].values
        
        self.input_size = len(column_list)

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)
        
    def aggr_over_days(self, n_days=1) -> pd.DataFrame:
        moods = self.train_df["mood"].values
        windowed_df = self.train_df.groupby(level=0).rolling(n_days).mean()
        windowed_df["mood"] = moods
        windowed_df = windowed_df.dropna()
        return windowed_df
        

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

def get_dataset_V1(batch_size, day_window=1):
    dataset = MoodDatasetV1("../data/train_data_v1.csv", day_window)
    trainset, testset = random_split(dataset, lengths=[round(0.7*len(dataset)),round(0.3*len(dataset))], generator=torch.Generator().manual_seed(155))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


dataset = MoodDatasetV1("../data/train_data_v1.csv", 1)

dataset.train_df

# %%
# TODO: check the values of mood when daywindow=1
# %%
