import torch
import math


class Polynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(() ))
        self.b = torch.nn.Parameter(torch.randn(() ))
        self.c = torch.nn.Parameter(torch.randn(() ))
        self.d = torch.nn.Parameter(torch.randn(() ))

    def forward(self, x):
        return self.a + self.b * x + self.c * x **2 + self.d * x **3

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'


x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)


model = Polynomial3()
print(model.a, model.b, model.c, model.d)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

for t in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    if t % 100 == 0:
        print(model.a, model.b, model.c, model.d)
        print(2 * (model.a * 2000 + model.b * x.sum() + model.c * x.pow(2).sum() + model.d * x.pow(3).sum() - y.sum()))
        print(2 * (model.a * x.sum() + model.b * x.pow(2).sum() + model.c * x.pow(3).sum() + model.d * x.pow(4).sum() - (y * x).sum()))
        print(2 * (model.a * x.pow(2).sum() + model.b * x.pow(3).sum() + model.c * x.pow(4).sum() + model.d * x.pow(5).sum() - (y * x.pow(2)).sum()))
        print(2 * (model.a * x.pow(3).sum() + model.b * x.pow(4).sum() + model.c * x.pow(5).sum() + model.d * x.pow(6).sum() - (y * x.pow(3)).sum()))
    optimizer.step()
    if t % 100 == 0:
        print(model.a, model.b, model.c, model.d)

print(f'Result: {model.string()}')
