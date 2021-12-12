import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len(data)
        return self.n_samples


class ToTensor:
    def __call__(self, samples):
        inputs, targets = samples
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MultiTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, samples):
        inputs, targets = samples
        inputs *= self.factor
        return inputs, targets


dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MultiTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))