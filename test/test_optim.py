import numpy as np
import torch

from easygrad.optim import SGD, Adam, RMSprop
from easygrad.tensor import Tensor

x_init = np.random.randn(1,3).astype(np.float32)
w_init = np.random.randn(3,3).astype(np.float32)
b_init = np.random.randn(1,3).astype(np.float32)


class EasyNet:
    def __init__(self):
        self.x = Tensor(x_init.copy())
        self.w = Tensor(w_init.copy())
        self.b = Tensor(b_init.copy())

    def forward(self):
        out = self.x.dot(self.w).relu()
        out = out.logsoftmax()
        out = out.mul(self.b).add(self.b).sum()
        return out

    def params(self):
        return [self.w, self.b]

class TorchNet:
    def __init__(self):
        self.x = torch.tensor(x_init.copy(), requires_grad=True)
        self.w = torch.tensor(w_init.copy(), requires_grad=True)
        self.b = torch.tensor(b_init.copy(), requires_grad=True)

    def forward(self):
        out = self.x.matmul(self.w).relu()
        out = torch.nn.functional.log_softmax(out, dim=1)
        out = out.mul(self.b).add(self.b).sum()
        return out

    def params(self):
        return [self.w, self.b]


def step_easygrad(optim, **kwargs):
    net = EasyNet()
    optimizer = optim(net.params(), **kwargs)
    out = net.forward()
    out.backward()
    optimizer.step()
    return [p.data for p in net.params()]

def step_torch(optim, **kwargs):
    net = TorchNet()
    optimizer = optim(net.params(), **kwargs)
    out = net.forward()
    out.backward()
    optimizer.step()
    return [p.detach().numpy() for p in net.params()]


class TestOptim:
    def test_sgd(self):
        params_easygrad = step_easygrad(SGD, lr=0.001)
        params_torch = step_torch(torch.optim.SGD, lr=0.001)
        for p1, p2 in zip(params_easygrad, params_torch):
            np.testing.assert_allclose(p1, p2, atol=1e-5)

    def test_rmsprop(self):
        params_easygrad = step_easygrad(RMSprop, lr=0.001, decay=0.99)
        params_torch = step_torch(torch.optim.RMSprop, lr=0.001, alpha=0.99)
        self.a = params_easygrad
        self.b = params_torch
        for p1, p2 in zip(params_easygrad, params_torch):
            np.testing.assert_allclose(p1, p2, atol=1e-5)

    def test_adam(self):
        params_easygrad = step_easygrad(Adam)
        params_torch = step_torch(torch.optim.Adam)
        self.a = params_easygrad
        self.b = params_torch
        for p1, p2 in zip(params_easygrad, params_torch):
            np.testing.assert_allclose(p1, p2, atol=1e-5)
