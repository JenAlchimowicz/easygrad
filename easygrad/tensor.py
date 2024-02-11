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

    def topological_sort(self):
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                if getattr(node, "ctx", False):
                    for child in node.ctx.children:
                        build_topo(child)
                    topo.append(node)
        build_topo(self)
        return topo

    def backward(self):
        assert self.data.size == 1, f"Can only backpropagate on arrays of size 1, got array shape {self.data.shape} and size {self.data.size}"

        self.grad = np.array([1])

        for node in reversed(self.topological_sort()):
            grads = node.ctx.op.backward(node.ctx, node.grad)
            if len(node.ctx.children) == 1:
                grads = [grads]
            for child, grad in zip(node.ctx.children, grads):
                assert(child.data.shape == grad.shape), f"Wrong shapes of gradients, data shape: {child.data.shape}, grad shape: {grad.shape}, op: {node.ctx.op}"
                child.grad = grad if child.grad is None else child.grad + grad

    # Non class ops (just use other functions)
    def mean(self):
        div = Tensor(np.array([1/self.data.size], dtype=self.data.dtype))
        return self.sum().mul(div)

    def __repr__(self):
        return f"Tensor of shape: {self.data.shape}, grad: {self.grad}, Data: {self.data}"


###### FUNCTION DEF #######

class Function:
    def apply(self, op, *tensors, **kwargs):  #self=Tensor, op(fixed), tensors=real input, kwargs=extra parameters e.g. shape in reshape
        inputs = [self] + list(tensors)
        ctx = Context(op, *inputs)
        out = Tensor(ctx.op.forward(ctx, *[t.data for t in inputs], **kwargs))
        out.ctx = ctx
        return out

def register(op, name):
    setattr(Tensor, name, partialmethod(op.apply, op))


###### ELEMENT WISE OPS #######

class Add(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        return x + y

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return grad, grad
register(Add, "add")

class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        return x - y

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return grad, -grad
register(Sub, "sub")

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


###### AGGREGATION OPS #######

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

class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, shape: tuple):
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        shape, = ctx.saved_for_backward
        return grad.reshape(shape)
register(Reshape, "reshape")


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

class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        out = 1 / (1 + np.power(np.e, -x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        out, = ctx.saved_for_backward
        return grad * out * (1 - out)
register(Sigmoid, "sigmoid")

class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        out = (2 / (1 + np.power(np.e, -2 * x))) - 1
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        out, = ctx.saved_for_backward
        return grad * (1 - np.square(out))
register(Tanh, "tanh")

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, x):
        max_per_sample = x.max(axis=1).reshape(-1,1)
        e_x = np.exp(x - max_per_sample)
        out = np.log(e_x / e_x.sum(axis=1).reshape(-1,1))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        out, = ctx.saved_for_backward
        return grad - np.exp(out)*grad.sum(axis=1).reshape((-1, 1))
register(LogSoftmax, "logsoftmax")
