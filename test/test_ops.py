import time

import numpy as np
import torch

from easygrad.tensor import Tensor


def helper_test_op(shapes, easygrad_fn, torch_fn, atol=1e-7, grad_atol=1e-7):
    torch_tensors = [torch.rand(shape, requires_grad=True) for shape in shapes]
    easy_tensors = [Tensor(tensor.detach().numpy()) for tensor in torch_tensors]
    
    # Forward test
    start = time.monotonic()
    out_easy = easygrad_fn(*easy_tensors)
    easy_fp = time.monotonic() - start

    start = time.monotonic()
    out_torch = torch_fn(*torch_tensors)
    torch_fp = time.monotonic() - start

    np.testing.assert_allclose(out_easy.data, out_torch.detach().numpy(), atol=atol)

    # Backward test
    start = time.monotonic()
    out_easy.mean().backward()
    easy_bp = time.monotonic() - start

    start = time.monotonic()
    out_torch.mean().backward()
    torch_bp = time.monotonic() - start

    for t1, t2 in zip(easy_tensors, torch_tensors):
        np.testing.assert_allclose(t1.data, t2.detach().numpy(), atol=grad_atol)
        np.testing.assert_allclose(t1.grad, t2.grad.detach().numpy(), atol=grad_atol)

    # Speed print
    print(f"FORWARD  | torch speed: {torch_fp*1000:.5f}, easygrad speed: {easy_fp*1000:.5f}, change: {(easy_fp - torch_fp) / torch_fp * 100:.2f}%")
    print(f"BACKWARD | torch speed: {torch_bp*1000:.5f}, easygrad speed: {easy_bp*1000:.5f}, change: {(easy_bp - torch_bp) / torch_bp * 100:.2f}%")


class TestOp:
    # Element wise ops
    def test_add(self):
        helper_test_op([(1, 16), (1, 16)], Tensor.add, lambda x,y: x+y)
        helper_test_op([(16, 32), (16, 32)], Tensor.add, lambda x,y: x+y)
    def test_sub(self):
        helper_test_op([(1, 16), (1, 16)], Tensor.sub, lambda x,y: x-y)
        helper_test_op([(16, 32), (16, 32)], Tensor.sub, lambda x,y: x-y)
    def test_mul(self):
        helper_test_op([(1, 16), (1, 16)], Tensor.mul, lambda x,y: x*y)
        helper_test_op([(16, 32), (16, 32)], Tensor.mul, lambda x,y: x*y)

    # Aggregation ops
    def test_sum(self):
        helper_test_op([(1, 16)], Tensor.sum, lambda x: x.sum(), atol=1e-6)
        helper_test_op([(16, 32)], Tensor.sum, lambda x: x.sum(), atol=1e-6)

    # Tensor ops
    def test_dot(self):
        helper_test_op([(1, 16), (16, 1)], Tensor.dot, lambda x,y: x.matmul(y))
        helper_test_op([(16, 32), (32, 16)], Tensor.dot, lambda x,y: x.matmul(y))
    def test_reshape(self):
        helper_test_op([(1, 16)], lambda x: x.reshape(shape=(4,4)), lambda x: torch.reshape(x, (4,4)))
        helper_test_op([(4, 4)], lambda x: x.reshape(shape=(1,16)), lambda x: torch.reshape(x, (1,16)))
        helper_test_op([(4, 3, 6, 6)], lambda x: x.reshape(shape=(-1,3,6,6)), lambda x: torch.reshape(x, (-1,3,6,6)))
    
    # Activation functions
    def test_relu(self):
        helper_test_op([(1, 16)], Tensor.relu, lambda x: x.relu())
        helper_test_op([(16, 32)], Tensor.relu, lambda x: x.relu())
    def test_sigmoid(self):
        helper_test_op([(1, 16)], Tensor.sigmoid, lambda x: x.sigmoid())
        helper_test_op([(16, 32)], Tensor.sigmoid, lambda x: x.sigmoid())
    def test_tanh(self):
        helper_test_op([(1, 16)], Tensor.tanh, lambda x: x.tanh(), atol=1e-6)
        helper_test_op([(16, 32)], Tensor.tanh, lambda x: x.tanh(), atol=1e-6)
    def test_logsoftmax(self):
        helper_test_op([(1, 16)], Tensor.logsoftmax, lambda x: torch.nn.functional.log_softmax(x, dim=1), atol=1e-6)
        helper_test_op([(16, 32)], Tensor.logsoftmax, lambda x: torch.nn.functional.log_softmax(x, dim=1), atol=1e-6)
