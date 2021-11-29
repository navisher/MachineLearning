import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
X = torch.rand(1, 28, 28, device=device)
logits = model(X)

pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab)
y_pred = pred_probab.argmax(1)

input_image = torch.rand(3, 28, 28)
flatten = nn.Flatten()
flat_image = flatten(input_image)

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
#print(hidden1.size())
#print(f"Before Relu: {hidden1}\n")
hidden1 = nn.ReLU()(hidden1)
#print(f"After Relu: {hidden1}")





