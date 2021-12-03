import torch
import math


a = torch.tensor([10, 1, 0.5])
b = torch.tensor([2, 3, 4])
print(a.sum())
print(torch.cuda.is_available())