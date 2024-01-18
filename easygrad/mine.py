from functools import partialmethod
import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            assert(False), "Only numpy arrays allowed as input to Tensor"
        self.data = data
        self.grad = None
        self.ctx = None


    def backward(self):
        global count
        count += 1
        print(count, self)
        print(self.ctx is None)
        if self.ctx is None:
            return
        
        if self.grad is None:
            self.grad = np.array([1])
        
        grads = self.ctx.op.backward(self.ctx, self.grad)
        if len(grads) == 1:
            grads = [grads]
        for child, grad in zip(self.ctx.children, grads):
            child.grad = grad
            child.backward()
                       
    def __repr__(self):
        return f"Tensor of shape: {self.data.shape}, grad: {self.grad}, Data: {self.data}"
        
        

class Context:
    def __init__(self, op, *tensors):
        self.children = tensors
        self.saved_for_backward = []
        self.op = op

    def save_for_backward(self, *arrays: np.ndarray):
        self.saved_for_backward.extend(arrays)


class Function:
    def apply(self, op, *tensors):  #self=Tensor, op(fixed), tensors=real input
        ctx = Context(op, self, tensors)
        out = Tensor(ctx.op.forward(ctx, self.data, *[t.data for t in tensors]))
        out.ctx = ctx
        return out

def register(op, name):
    setattr(Tensor, name, partialmethod(op.apply, op))


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_for_backward
        return grad*y, grad*x
register(Mul, "mul")



count = 0

a = Tensor(np.arange(3))
b = Tensor(np.arange(3)+5)
c = a.mul(b)
c.backward()
