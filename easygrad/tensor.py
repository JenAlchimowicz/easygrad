from functools import partialmethod

import numpy as np


class Context:
    def __init__(self, op, *tensors):
        self.children = tensors
        self.saved_for_backward = []
        self.op = op

    def save_for_backward(self, *arrays: np.ndarray):
        self.saved_for_backward.extend(arrays)


class Tensor:
    def __init__(self, data: np.ndarray):
        assert(isinstance(data, np.ndarray)), "Only numpy arrays allowed as input to Tensor"
        self.data = data
        self.grad = None
        self.ctx = None

    def backward(self):
        if self.ctx is None:  # Nothing to backpropagate to
            return
        
        if self.grad is None:
            self.grad = np.array([1])
        
        grads = self.ctx.op.backward(self.ctx, self.grad)
        if len(self.ctx.children) == 1:
            grads = [grads]
        for child, grad in zip(self.ctx.children, grads):
            assert(child.data.shape == grad.shape), f"Wrong shapes of gradients, data shape: {child.data.shape}, grad shape: {grad.shape}, op: {self.ctx.op}"
            child.grad = grad
            child.backward()

    # Non class ops (just use other functions)
    def mean(self):
        div = Tensor(np.array([1/self.data.size], dtype=self.data.dtype))
        return self.sum().mul(div)
                       
                       
    def __repr__(self):
        return f"Tensor of shape: {self.data.shape}, grad: {self.grad}, Data: {self.data}"


###### FUNCTION DEF #######

class Function:
    def apply(self, op, *tensors):  #self=Tensor, op(fixed), tensors=real input
        inputs = [self] + list(tensors)
        ctx = Context(op, *inputs)
        out = Tensor(ctx.op.forward(ctx, *[t.data for t in inputs]))
        out.ctx = ctx
        return out

def register(op, name):
    setattr(Tensor, name, partialmethod(op.apply, op))


###### SIMPLE OPS #######
    
class Add(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        return x + y

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return grad, grad
register(Add, "add")

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, y = ctx.saved_for_backward
        return grad*y, grad*x
register(Mul, "mul")

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        ctx.save_for_backward(x)
        return np.array([x.sum()])

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, = ctx.saved_for_backward
        return np.ones_like(x) * grad
register(Sum, "sum")


###### TENSOR OPS #######

class Dot(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return x.dot(y)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, y = ctx.saved_for_backward
        # print(x.shape, y.shape, grad.shape)
        x_grad = grad.dot(y.T)
        y_grad = x.T.dot(grad)
        return x_grad, y_grad
register(Dot, "dot")


###### ACTIVATIONS #######

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        ctx.save_for_backward(x)
        return np.maximum(x, 0)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, = ctx.saved_for_backward
        grad_out = grad.copy()
        grad_out[x < 0] = 0
        return grad_out
register(ReLU, "relu")

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, x):
        e_x = np.exp(x - x.max())
        out = np.log(e_x / e_x.sum())
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        output, = ctx.saved_for_backward
        return grad - np.exp(output)*grad.sum(axis=1).reshape((-1, 1))
register(LogSoftmax, "logsoftmax")

