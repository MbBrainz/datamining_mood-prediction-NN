#%%
from matplotlib import pyplot as plt
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
features = 15 # This is without circumplex
in_features = features
EPOCHS = 10000
BACHSIZE = 100
N_TRAIN_EXAMPLES = 100 * 3
N_TEST_EXAMPLES = 10 * 3

def define_model(trial):
    
    #internal hyper parameters
    n_layers =  trial.suggest_int("n_layers", 1, 30)
    out_features = trial.suggest_int(f"n_neurons", 8, 128)
    p = trial.suggest_float("dropout", 0.2, 0.5)
    
    global features
    in_features = features
    
    layers = []
    for i in range(n_layers):
        # HP: optimize number of neurons per layer
        # out_features = trial.suggest_int(f"n_units_l{i}", 8, 128)
        # p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        layers.append(nn.Dropout(p))
        
        in_features = out_features
    
    layers.append(torch.nn.Linear(in_features, features))
    layers.append(torch.nn.Linear(features, 1))
    
    return nn.Sequential(*layers)
    
# This function describes how optuna will test the model
def objective(trial: optuna.Trial):
    ## hyperparameters outside model
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    mse_loss = nn.MSELoss()
    # TODO: check R^2 method
    day_window = trial.suggest_int("day_window", 1, 10)
    
    model = define_model(trial).to(DEVICE)
    train_loader, test_loader = get_dataset_V1(BACHSIZE, day_window=day_window)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    
    for epoch in range(EPOCHS):
        
        # starts training for this EPOCH
        model.train() #just sets the state of the object to training mode
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * BACHSIZE >= N_TRAIN_EXAMPLES:
                break
            
            data, target = data.to(DEVICE), target.unsqueeze(1).to(DEVICE)
    
            optimizer.zero_grad()
            output = model(data)
            loss = mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        
        # starts the evaluation mode of the model  for this epoch
        model.eval()
        correct = 0
        losses = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # # Limiting validation data.
                if batch_idx * BACHSIZE >= N_TEST_EXAMPLES:
                    break
                data, target = data.to(DEVICE), target.unsqueeze(1).to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                output = model(data)
                loss = mse_loss(output, target)
                
                losses.append(loss)
                # pred = output.argmax(dim=1, keepdim=True)
                # correct += pred.eq(target.view_as(pred)).sum().item()

        # accuracy = correct / min(len(test_loader.dataset), N_TEST_EXAMPLES)
        mean_loss = np.mean(losses)
        trial.report(mean_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mean_loss

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
optuna.visualization.plot_contour(study,["n_layers", "day_window"])

importances = optuna.importance.get_param_importances(study)
print(importances)

plt.bar(importances.keys(),importances.values())
# %%
# ------------------- Temporal NN (RNN or LSTM) ----------------------
# TODO: implement