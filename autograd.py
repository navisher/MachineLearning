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
print(f"Gradient function for z: {z.grad_fn}")
print(f"Gradient function for loss: {loss.grad_fn}")
loss.backward()
print(w.grad)
print(b.grad)

# inp = torch.eye(5, requires_grad=True)
# out = (inp+2).pow(2)
# out.backward(torch.ones_like(inp), retain_graph=True)
# print("First call\n", inp.grad)
# out.backward(torch.ones_like(inp), retain_graph=True)
# print("\nSecond call\n", inp.grad)
# inp.grad.zero_()
# out.backward(torch.ones_like(inp), retain_graph=True)
# print("\nCall after zeroing gradients\n", inp.grad)

