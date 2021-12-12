import torch
import torch.nn as nn
import numpy as np


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted), axis=0)
    return loss


Y = np.array([1, 0, 0])

y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)
#print(l1, l2)

# CrossEntropyLoss下， 实际值Y只用序号表示目标所属的类别
# 比如前例中 Y = np.array([1, 0, 0])会被表示为 Y = torch.tensor([0])
loss = nn.CrossEntropyLoss()
Y = torch.tensor([2, 0, 1])
y_pred_good = torch.tensor([[0.1, 1.0, 10.0],
                            [2.0, 0.0, 0.1],
                            [2.0, 100.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3],
                           [0.5, 2.0, 0.3],
                           [0.5, 2.0, 0.3]])
print(torch.softmax(y_pred_good, dim=1))
print(torch.softmax(y_pred_bad, dim=1))

l3 = loss(y_pred_good, Y)
l4 = loss(y_pred_bad, Y)
print(l3.item(), l4.item())

_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)
print(predictions1, predictions2)
