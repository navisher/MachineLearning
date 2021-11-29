import torch
import numpy as np

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
print(w)
b = torch.randn(3, requires_grad=True)
print(b)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# print(f"Gradient function for z: {z.grad_fn}")
# print(f"Gradient function for loss: {loss.grad_fn}")
loss.backward()
print(w.grad)
print(b.grad)


