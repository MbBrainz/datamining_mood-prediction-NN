import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd

#TODO: The dataset is now features and targets of the same dat in the same row
# This should be changed to one of these:
# - Sum 5 consequtive days and ad the mood of the rest
# - put the mood of the next day on the previous row ( shape will be N-1 )
class MoodDatasetV1(Dataset):
    
    def __init__(self,file_name , train=True,):
        train_df=pd.read_csv(file_name)
        target = "mood"
        column_list = train_df.columns.tolist()
        column_list.remove(target)
        column_list.remove("id")
        column_list.remove("date")

        x=train_df[column_list].values
        y=train_df[target].values
        
        self.input_size = len(column_list)

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)
        

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

def get_dataset_V1(batch_size):
    dataset = MoodDatasetV1("../data/train_data_v1.csv")
    trainset, testset = random_split(dataset, lengths=[round(0.7*len(dataset)),round(0.3*len(dataset))], generator=torch.Generator().manual_seed(155))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader