#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd


#%%

class MyDataset(Dataset):
    
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

def get_dataset(batch_size):
    dataset = MyDataset("../data/train_data_v1.csv")
    trainset, testset = random_split(dataset, lengths=[round(0.7*len(dataset)),round(0.3*len(dataset))], generator=torch.Generator().manual_seed(155))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# %%



#%%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} DEVICE")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits





# %%
criterion = nn.MSELoss()
train_loader, test_loader = get_dataset(10)
model = NeuralNetwork(input_size=16).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

num_epochs = 1
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        
        print(data.shape)
        
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        print(f"[Epoch{epoch} batch{batch_idx}] loss = {loss}")
        


        
# %%
# optimizing hyperparameters using optuna https://optuna.org/#code_examples 
import optuna


# possible optimizers"
features = 16
in_features = features
EPOCHS = 1
BACHSIZE = 100
N_TRAIN_EXAMPLES = 100
N_TEST_EXAMPLES = 10

def define_model(trial):
    n_layers =  trial.suggest_int("n_layers", 1, 10)
    global features
    in_features = features
    layers = []
    for i in range(n_layers):
        # HP: optimize number of neurons per layer
        out_features = trial.suggest_int(f"n_units_l{i}", features, 128)
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    
    layers.append(torch.nn.Linear(in_features, features))
    layers.append(torch.nn.Linear(features, 1))
    
    return nn.Sequential(*layers)
    

def objective(trial: optuna.Trial):
    ### Our code
    ## hyperparameters outside model
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    model = define_model(trial).to(DEVICE)
    train_loader, test_loader = get_dataset(BACHSIZE)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # if batch_idx * BACHSIZE >= N_TRAIN_EXAMPLES:
            #     break
                
            data, target = data.to(DEVICE), target.to(DEVICE)
            
    
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss(output, target)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                # Limiting validation data.
                if batch_idx * BACHSIZE >= N_TEST_EXAMPLES:
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(test_loader.dataset), N_TEST_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

#%%

from optuna.trial import TrialState
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, timeout=600)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%

# %%
