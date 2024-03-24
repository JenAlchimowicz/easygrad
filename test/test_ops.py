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
        helper_test_op([(1,16), (1,16)], Tensor.add, lambda x,y: x+y)
        helper_test_op([(16,32), (16,32)], Tensor.add, lambda x,y: x+y)
    def test_sub(self):
        helper_test_op([(1,16), (1,16)], Tensor.sub, lambda x,y: x-y)
        helper_test_op([(16,32), (16,32)], Tensor.sub, lambda x,y: x-y)
    def test_mul(self):
        helper_test_op([(1,16), (1,16)], Tensor.mul, lambda x,y: x*y)
        helper_test_op([(16,32), (16,32)], Tensor.mul, lambda x,y: x*y)
    def test_div(self):
        helper_test_op([(1,16), (1,16)], Tensor.div, lambda x,y: x/y, atol=1e-4)
        helper_test_op([(16,32), (16,32)], Tensor.div, lambda x,y: x/y, atol=1e-4)
    def test_sqrt(self):
        helper_test_op([(1,16)], Tensor.sqrt, lambda x: x.sqrt(), atol=1e-6)
        helper_test_op([(16,32)], Tensor.sqrt, lambda x: x.sqrt(), atol=1e-6)

    # Reduce ops
    def test_sum(self):
        helper_test_op([(1,16)], Tensor.sum, lambda x: x.sum(), atol=1e-5)
        helper_test_op([(16,32)], Tensor.sum, lambda x: x.sum(), atol=1e-5)
        helper_test_op([(16)], lambda x: x.sum(axis=0), lambda x: x.sum(axis=0), atol=1e-5)
        # helper_test_op([(1,16)], lambda x: x.sum(axis=1), lambda x: x.sum(axis=1), atol=1e-5)  # TODO: make output shapes consistent with PyTorch if needed
        helper_test_op([(2,16)], lambda x: x.sum(axis=1), lambda x: x.sum(axis=1), atol=1e-5)
        helper_test_op([(4,4,4)], lambda x: x.sum(axis=2), lambda x: x.sum(axis=2), atol=1e-5)
        helper_test_op([(4,4,4)], lambda x: x.sum(axis=(1,2)), lambda x: x.sum(axis=(1,2)), atol=1e-5)
        helper_test_op([(4,1,4)], lambda x: x.sum(axis=(2)), lambda x: x.sum(axis=(2)), atol=1e-5)
    def test_sum_with_keepdims(self):
        helper_test_op([(4,4,4)], lambda x: x.sum(axis=2, keepdims=True), lambda x: x.sum(axis=2, keepdim=True), atol=1e-5)
        helper_test_op([(4,4,4)], lambda x: x.sum(axis=(1,2), keepdims=True), lambda x: x.sum(axis=(1,2), keepdim=True), atol=1e-5)
        helper_test_op([(4,1,4)], lambda x: x.sum(axis=(2), keepdims=True), lambda x: x.sum(axis=(2), keepdim=True), atol=1e-5)
    def test_mean(self):
        helper_test_op([(1,16)], Tensor.mean, lambda x: x.mean(), atol=1e-6)
        helper_test_op([(16,32)], Tensor.mean, lambda x: x.mean(), atol=1e-6)
        helper_test_op([(16)], lambda x: x.mean(axis=0), lambda x: x.mean(axis=0), atol=1e-6)
        # helper_test_op([(1,16)], lambda x: x.mean(axis=1), lambda x: x.mean(axis=1), atol=1e-6)
        helper_test_op([(2,16)], lambda x: x.mean(axis=1), lambda x: x.mean(axis=1), atol=1e-6)
        helper_test_op([(4,4,4)], lambda x: x.mean(axis=2), lambda x: x.mean(axis=2), atol=1e-6)
        helper_test_op([(4,4,4)], lambda x: x.mean(axis=(1,2)), lambda x: x.mean(axis=(1,2)), atol=1e-6)
        helper_test_op([(4,1,4)], lambda x: x.mean(axis=(2)), lambda x: x.mean(axis=(2)), atol=1e-6)
    def test_mean_with_keepdims(self):
        helper_test_op([(4,4,4)], lambda x: x.mean(axis=2, keepdims=True), lambda x: x.mean(axis=2, keepdim=True), atol=1e-6)
        helper_test_op([(4,4,4)], lambda x: x.mean(axis=(1,2), keepdims=True), lambda x: x.mean(axis=(1,2), keepdim=True), atol=1e-6)
        helper_test_op([(4,1,4)], lambda x: x.mean(axis=(2), keepdims=True), lambda x: x.mean(axis=(2), keepdim=True), atol=1e-6)

    # Tensor ops
    def test_dot(self):
        helper_test_op([(1,16), (16, 1)], Tensor.dot, lambda x,y: x.matmul(y))
        helper_test_op([(16,32), (32, 16)], Tensor.dot, lambda x,y: x.matmul(y), atol=1e-5)
        helper_test_op([(1,1,1,5), (1,1,5,1)], Tensor.dot, lambda x,y: x.matmul(y))
        helper_test_op([(2,3,4,5), (2,3,5,6)], Tensor.dot, lambda x,y: x.matmul(y))
        helper_test_op([(3,4,5), (2,3,5,6)], Tensor.dot, lambda x,y: x.matmul(y))
    def test_reshape(self):
        helper_test_op([(1,16)], lambda x: x.reshape(shape=(4,4)), lambda x: torch.reshape(x, (4,4)))
        helper_test_op([(4,4)], lambda x: x.reshape(shape=(1,16)), lambda x: torch.reshape(x, (1,16)))
        helper_test_op([(4,3,6,6)], lambda x: x.reshape(shape=(-1,3,6,6)), lambda x: torch.reshape(x, (-1,3,6,6)))
    def test_expand(self):
        helper_test_op([(1)], lambda x: x.expand(shape=(2)), lambda x: x.expand(2))
        helper_test_op([(1)], lambda x: x.expand(shape=(2,2)), lambda x: x.expand(2,2))
        helper_test_op([(2)], lambda x: x.expand(shape=(3,4,2)), lambda x: x.expand(3,4,2))
        helper_test_op([(2,2)], lambda x: x.expand(shape=(4,3,2,2)), lambda x: x.expand(4,3,2,2))
        helper_test_op([(2,1)], lambda x: x.expand(shape=(2,2)), lambda x: x.expand(2,2), atol=1e-5)
        helper_test_op([(2,1,1)], lambda x: x.expand(shape=(2,3,4)), lambda x: x.expand(2,3,4), atol=1e-5)
        # NOTE: expand function is not supposed to handle the below case (same in PyTorch). Use add_dims + expand to achieve that.
        # helper_test_op([(2,2)], lambda x: x.expand(shape=(2,2,3,4)), lambda x: x.expand(2,2,3,4))
    def test_permute(self):
        helper_test_op([(2,3)], lambda x: x.permute(dims=(1,0)), lambda x: x.permute(1,0))
        helper_test_op([(1,2,3,4)], lambda x: x.permute(dims=(0,3,1,2)), lambda x: x.permute(0,3,1,2))
        helper_test_op([(1,2,3,4,5)], lambda x: x.permute(dims=(4,0,1,3,2)), lambda x: x.permute(4,0,1,3,2))
    
    # Activation functions
    def test_relu(self):
        helper_test_op([(1,16)], Tensor.relu, lambda x: x.relu())
        helper_test_op([(16,32)], Tensor.relu, lambda x: x.relu())
    def test_gelu_original(self):
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/activations.py
        def gelu_original_transformers(input):
            return input * 0.5 * (1.0 + torch.erf(input / np.sqrt(2.0)))
        helper_test_op([(1,16)], Tensor.gelu_original, lambda x: gelu_original_transformers(x), atol=1e-6)
        helper_test_op([(16,32)], Tensor.gelu_original, lambda x: gelu_original_transformers(x), atol=1e-6)
    def test_gelu(self):
        helper_test_op([(1,16)], Tensor.gelu, lambda x: torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-5)
        helper_test_op([(16,32)], Tensor.gelu, lambda x: torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-5)
    def test_sigmoid(self):
        helper_test_op([(1,16)], Tensor.sigmoid, lambda x: x.sigmoid())
        helper_test_op([(16,32)], Tensor.sigmoid, lambda x: x.sigmoid())
    def test_tanh(self):
        helper_test_op([(1,16)], Tensor.tanh, lambda x: x.tanh(), atol=1e-6)
        helper_test_op([(16,32)], Tensor.tanh, lambda x: x.tanh(), atol=1e-6)
    def test_logsoftmax(self):
        helper_test_op([(1,16)], Tensor.logsoftmax, lambda x: torch.nn.functional.log_softmax(x, dim=1), atol=1e-6)
        helper_test_op([(16,32)], Tensor.logsoftmax, lambda x: torch.nn.functional.log_softmax(x, dim=1), atol=1e-6)
    def test_softmax(self):
        helper_test_op([(1,16)], lambda x: x.softmax(dim=1), lambda x: torch.nn.functional.softmax(x, dim=1), atol=1e-4)
        helper_test_op([(4,8,16)], lambda x: x.softmax(dim=1), lambda x: torch.nn.functional.softmax(x, dim=1), atol=1e-4)
        helper_test_op([(4,8,16,32)], lambda x: x.softmax(dim=3), lambda x: torch.nn.functional.softmax(x, dim=3), atol=1e-4)
