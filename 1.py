import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_sample = xy.shape[0]


    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(data)
        return self.n_sample


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_interations = math.ceil(total_samples/4)
print(n_interations, total_samples)

for epoch in range(num_epochs):
    for i, (input, labels) in enumerate(dataloader):
        # forward backward
        if (i+1)%5 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step {i+1}/{n_interations}, inputs {input.shape}')

torchvision.datasets.MNIST()
# fasion-mnist, cifar, coco