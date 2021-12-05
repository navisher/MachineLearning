from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")

#print(x_train.shape)

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
bs = 64
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs*2)
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())

import math
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

# def model(xb):
#     return log_softmax(xb @ weights + bias)



# xb = x_train[0:bs]
# preds = model(xb)
#print(preds[0], preds.shape)


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


#loss_func = nll
y_b = y_train[0:bs]

#loss = loss_func(preds, y_b)
#print(y_b)
#print(loss)


def accuracy(output, yb):
    index = torch.argmax(output, 1)
    return (index == yb).float().mean()


#from IPython.core.debugger import set_trace
lr = 0.5
epochs = 2


import torch.nn.functional as F
loss_func = F.cross_entropy


# def model(xb):
#     return xb @ weights + bias



# for epoch in range(epochs):
#     for i in range((n - 1) // bs + 1):
#         start = i * bs
#         end = start + bs
#         xb = x_train[start:end]
#         yb = y_train[start:end]
#         pred = model(xb)
#
#         loss = loss_func(pred, yb)
#
#         loss.backward()
#
#         with torch.no_grad():
#             weights -= weights.grad * lr
#             bias -= bias.grad * lr
#             weights.grad.zero_()
#             bias.grad.zero_()
#
#     print(loss_func(model(xb), yb), accuracy(model(xb), yb))

from torch import nn


class Minist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

model = Minist_Logistic()

def get_model():
    model = Minist_Logistic()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    return model, opt

def fit():
    model, opt = get_model()
    for epoch in range(epochs):
        for batch in train_dl:
            xb, yb = batch
            loss = loss_func(model(xb), yb)
            loss.backward()

            opt.step()
            model.zero_grad()

        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

        print(epoch, valid_loss / len(valid_dl))
fit()



