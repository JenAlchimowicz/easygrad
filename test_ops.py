import numpy as np
import torch

from easygrad.tensor import Tensor

def helper_test_op(shapes, easygrad_fn, torch_fn, atol=1e-7, grad_atol=1e-7):
    torch_tensors = [torch.rand(shape, requires_grad=True) for shape in shapes]
    easy_tensors = [Tensor(tensor.detach().numpy()) for tensor in torch_tensors]
    
    # Forward test
    out_easy = easygrad_fn(*easy_tensors)
    out_torch = torch_fn(*torch_tensors)
    np.testing.assert_allclose(out_easy, out_torch, atol=atol)

    # Backward test
    out_easy.mean().backward()
    out_torch.mean().backward()
    for t1, t2 in zip(easy_tensors, torch_tensors):
        np.testing.assert_allclose(t1, t2, atol=grad_atol)

    # Speed test
    

class TestOp:
    binary_simple_shapes1 = 
    binary_simple_shapes2 = [(16, 32), (16,32)]

    unary_simple_shapes1 = [(1, 16)]
    unary_simple_shapes2 = [(16, 32)]
    
    # binary_simple_shapes1 = [(1, 16), (1,16)]
    # binary_simple_shapes2 = [(16, 32), (16,32)]
    
    def test_add(self):
        helper_test_op([(1, 16), (1, 16)], Tensor.add, lambda x,y: x+y)
        helper_test_op([(16, 32), (16, 32)], Tensor.add, lambda x,y: x+y)
    def test_sub(self):
        pass
    def test_mul(self):
        helper_test_op([(1, 16), (1, 16)], Tensor.mul, lambda x,y: x*y)
        helper_test_op([(16, 32), (16, 32)], Tensor.mul, lambda x,y: x*y)
    def test_sum(self):
        pass
    def test_dot(self):
        pass
    def test_softmax(self):
        pass
    def test_sigmoid(self):
        pass
    def test_tanh(self):
        pass