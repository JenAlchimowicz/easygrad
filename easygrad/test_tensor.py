import numpy as np
import torch

from tensor import Tensor


def test_tensor_backward():
    a = np.random.rand(1, 5).astype(np.float32)
    b = np.random.rand(5, 3).astype(np.float32)
    c = np.random.rand(1, 3).astype(np.float32)

    a1, b1, c1 = Tensor(a), Tensor(b), Tensor(c)
    d1 = a1.dot(b1).relu() #1x3
    e1 = d1.add(c1).mul(c1)
    f1 = e1.logsoftmax()
    g1 = f1.sum()
    g1.backward()
    result_easy = [g1.data, a1.grad, b1.grad]

    a2 = torch.tensor(a, requires_grad=True)
    b2 = torch.tensor(b, requires_grad=True)
    c2 = torch.tensor(c, requires_grad=True)
    d2 = a2.matmul(b2).relu() #3x1
    e2 = d2.add(c2).mul(c2)
    f2 = torch.nn.functional.log_softmax(e2, dim=1)
    g2 = f2.sum()
    g2.backward()
    result_torch = [g2.detach().numpy(), a2.grad.detach().numpy(), b2.grad.detach().numpy()]

    for easy, torch_ in zip(result_easy, result_torch):
        np.testing.assert_allclose(easy, torch_, atol=1e-5)
