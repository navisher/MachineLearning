# 1) Design model ( input, output size, forward pass)
# 2) Constrct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass : gradient
#   - update weight

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare datas
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
# 1) model
class LogisticRegression(nn.Module):
    def __init__(self, input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.02
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 10000
for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)% 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')





#------------linear regression-----------
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
#
# # 0) prepare data
# X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
#
# X = torch.from_numpy(X_numpy.astype(np.float32))
# y = torch.from_numpy(y_numpy.astype(np.float32))
# y = y.view(y.shape[0], 1)
# n_samples, n_features = X.shape
#
# # 1) model
# input_size = n_features
# output_size = 1
# model = nn.Linear(input_size, output_size)
#
# # 2)loss and optimizer
# learning_rate = 0.01
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# # 3)training loops
# num_epochs = 100
# for epoch in range(num_epochs):
#     # forward pass and loss
#     y_pred = model(X)
#     loss = criterion(y_pred, y)
#
#     # backward
#     loss.backward()
#
#     # update
#     optimizer.step()
#     optimizer.zero_grad()
#
#     if (epoch + 1) % 10 == 0:
#         print(f'epoch+1: {epoch+1}, loss = {loss.item():.4f}')
#
# # plot
# predicted = model(X).detach()
# plt.plot(X_numpy, y_numpy, 'ro')
# plt.plot(X_numpy, predicted, 'b')
# plt.show()




#----------------
# f = w * y

# f = 2 * x
# X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
# Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
#
# X_test = torch.tensor([5], dtype=torch.float32)
#
# n_samples, n_features = X.shape
# print(n_samples, n_features)
#
# input_size = n_features
# output_size = n_features
# #model = nn.Linear(input_size, output_size)
#
# class LinearRegression(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegression, self).__init__()
#         self.lin = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         return self.lin(x)
#
# model = LinearRegression(input_size, output_size)
#
# print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
#
# learning_rate = 0.01
# n_iters = 1000
#
# loss = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
#
#
# for epoch in range(n_iters):
#     #prediction
#     y_pred = model(X)
#
#     #loss
#     l = loss(Y, y_pred)
#
#     #gradient = backward pass
#     l.backward()
#
#     optimizer.step()
#
#     optimizer.zero_grad()
#     if epoch % 10 == 0:
#         [w, b] = model.parameters()
#         print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
# print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')