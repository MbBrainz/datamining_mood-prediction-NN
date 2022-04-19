#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import optuna

from MoodDataset import get_dataset_V1, MoodDatasetV1
# %% [markdown]
# ## Hyper parameter optimisation
# In the cells below we test a set of hyper parameters (where you see `trial.suggest_...`) and we find the optimal value to fully train the model with
# 
# The package optuna is light weight and helps us test multiple parameters.
# 
# 
# #
# %%
# ------------------- NN (NON-TEMPORAL) ----------------------
# optimizing hyperparameters using optuna https://optuna.org/#code_examples 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} DEVICE")

# possible optimizers"
features = 16 # This is without circumplex
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
        # TODO: discuss network layer design
        out_features = trial.suggest_int(f"n_units_l{i}", features, 128)
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    
    layers.append(torch.nn.Linear(in_features, features))
    layers.append(torch.nn.Linear(features, 1))
    
    return nn.Sequential(*layers)
    
# This function describes how optuna will test the model
def objective(trial: optuna.Trial):
    ## hyperparameters outside model
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    mse_loss = nn.MSELoss()
    
    model = define_model(trial).to(DEVICE)
    train_loader, test_loader = get_dataset_V1(BACHSIZE)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * BACHSIZE >= N_TRAIN_EXAMPLES:
                break
            
            data, target = data.to(DEVICE), target.unsqueeze(1).to(DEVICE)
    
            optimizer.zero_grad()
            output = model(data)
            loss = mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # # Limiting validation data.
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
# ------------------- Temporal NN (RNN or LSTM) ----------------------
# TODO: implement