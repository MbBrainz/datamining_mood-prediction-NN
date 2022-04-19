"""
This file defines a basic neural network and provides a simple train function.

It still needs:
# TODO: Proper quality assessment using test data (Confidence interval etc)


"""

#%%
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

from MoodDataset import get_dataset_V1

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} DEVICE")

criterion = nn.MSELoss()
train_loader, test_loader = get_dataset_V1(10)
model = NeuralNetwork(input_size=16).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

num_epochs = 1
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device=DEVICE)
        targets = targets.unsqueeze(1).to(device=DEVICE) 
        
        print(data.shape)
        
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        print(f"[Epoch{epoch} batch{batch_idx}] loss = {loss}")
        




# This is purely copied code, nice example code
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )
#         self.inputSize = 3
#         self.outputSize = 1
#         self.hiddenSize = 3

#         self.W1 = np.random.rand(self.inputSize, self.hiddenSize)
#         self.W2 = np.random.rand(self.hiddenSize, self.outputSize)

#         self.error_list = []
#         self.limit = 0.5
#         self.true_positives = 0
#         self.false_positives = 0
#         self.true_negatives = 0
#         self.false_negatives = 0

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

#     def sigmoid(self, s):
#         return 1 / (1 + np.exp(-s))

#     def sigmoidPrime(self, s):
#         return s * (1 - s)

#     def backward(self, X, y, o):
#         self.o_error = y - o
#         self.o_delta = self.o_error * self.sigmoidPrime(o)
#         self.z2_error = np.matmul(self.o_delta,
#                                   np.matrix.transpose(self.W2))
#         self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
#         self.W1 += np.matmul(np.matrix.transpose(X), self.z2_delta)
#         self.W2 += np.matmul(np.matrix.transpose(self.z2),
#                              self.o_delta)

#     def train(self, X, y, epochs):
#         for epoch in range(epochs):
#             o = self.forward(X)
#             self.backward(X, y, o)
#             self.error_list.append(np.abs(self.o_error).mean())

#     def predict(self, x_predicted):
#         return self.forward(x_predicted).item()

#     def view_error_development(self):
#         plt.plot(range(len(self.error_list)), self.error_list)
#         plt.title('Mean Sum Squared Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')

#     def test_evaluation(self, input_test, output_test):
#         for i, test_element in enumerate(input_test):
#             if self.predict(test_element) > self.limit and \
#                     output_test[i] == 1:
#                 self.true_positives += 1
#             if self.predict(test_element) < self.limit and \
#                     output_test[i] == 1:
#                 self.false_negatives += 1
#             if self.predict(test_element) > self.limit and \
#                     output_test[i] == 0:
#                 self.false_positives += 1
#             if self.predict(test_element) < self.limit and \
#                     output_test[i] == 0:
#                 self.true_negatives += 1
#         print('True positives: ', self.true_positives,
#               '\nTrue negatives: ', self.true_negatives,
#               '\nFalse positives: ', self.false_positives,
#               '\nFalse negatives: ', self.false_negatives,
#               '\nAccuracy: ',
#               (self.true_positives + self.true_negatives) /
#               (self.true_positives + self.true_negatives +
#                self.false_positives + self.false_negatives))
        