import torch
import math


class LegendrePolynominal3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)


def P2(x):
    return 1.5 * (5.0 * x ** 2.0 - 1.0)

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device)
y = torch.sin(x)
learning_rate = 5e-6

a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

for t in range(2000):
    P3 = LegendrePolynominal3.apply
    pred_y = a + b * P3(c + d * x)
    #pred_y = a + b * x + c * x ** 2 + d * x ** 3
    loss = (pred_y - y).pow(2).sum()
    # if t % 100 == 0:
    #     print(t, loss.item())

    loss.backward()
    # grad_y_pred = 2 * (pred_y - y)
    # grad_a = grad_y_pred.sum()
    # grad_b = (grad_y_pred * x).sum()
    # grad_c = (grad_y_pred * x ** 2).sum()
    # grad_d = (grad_y_pred * x ** 3).sum()
    grad_y_pred = 2.0 * (pred_y - y)

    with torch.no_grad():
        if t % 100 == 0:
            print(a.grad, b.grad, c.grad, d.grad)
            print(grad_y_pred.sum(), (grad_y_pred * P3(c + d * x)).sum(),
                  (grad_y_pred * b * P2(c + d * x)).sum(), (grad_y_pred * b * P2(c + d * x) * x).sum())
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad


        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)")
