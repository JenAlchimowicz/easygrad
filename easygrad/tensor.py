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
        assert(isinstance(data, np.ndarray)), f"Only numpy arrays allowed as input to Tensor, got {type(data)}"
        self.data = data
        self.grad = None
        self.ctx = None
        self.training = True

        self.nnode = 0
        self.nloop = 0

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
            self.nnode += 1
            grads = node.ctx.op.backward(node.ctx, node.grad)
            if len(node.ctx.children) == 1:
                grads = [grads]
            for child, grad in zip(node.ctx.children, grads):
                self.nloop += 1
                assert(child.data.shape == grad.shape), f"Wrong shapes of gradients, data shape: {child.data.shape}, grad shape: {grad.shape}, op: {node.ctx.op}"
                child.grad = grad if child.grad is None else child.grad + grad

    # Non class ops (just use other functions)
    def mean(self, axis=None, keepdims=False):
        out = self.sum(axis=axis, keepdims=keepdims)
        div = np.ones(out.shape) * (np.prod(out.shape) / np.prod(self.shape))
        div = Tensor(div)
        return out.mul(div)

    def square(self):
        return self.mul(self)
    
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor of shape: {self.data.shape}, grad: {self.grad}, Data: {self.data}"
    
    def __add__(self, x):
        return self.add(x)
    def __sub__(self, x):
        return self.sub(x)
    def __mul__(self, x):
        return self.mul(x)
    def __truediv__(self, x):
        return self.div(x)


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

class Div(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return x / y

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, y = ctx.saved_for_backward
        return grad*(1/y), grad*(-x * np.power(y, -2))
register(Div, "div")

class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        ctx.save_for_backward(x)
        return np.sqrt(x)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, = ctx.saved_for_backward
        return grad * (0.5 * np.power(x, -0.5))
register(Sqrt, "sqrt")

class Log(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, = ctx.saved_for_backward
        return grad * 1 / x
register(Log, "log")

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        out = np.exp(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        out, = ctx.saved_for_backward
        return grad * out
register(Exp, "exp")


###### REDUCE OPS #######

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, axis=None, keepdims=False):
        if not axis:
            axis = tuple(range(len(x.shape)))
            axis_ = axis[1:]
        else:
            axis_ = axis
        ctx.save_for_backward(x.shape, axis_, keepdims)
        out = x.sum(axis=axis, keepdims=keepdims)
        return np.array([out]) if out.size == 1 else out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        original_shape, axis, keepdims = ctx.saved_for_backward
        if not keepdims:
            grad = np.expand_dims(grad, axis)
        grad = np.broadcast_to(grad, original_shape)
        return grad
register(Sum, "sum")


###### TENSOR OPS #######

class Dot(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return np.matmul(x, y)

    # TODO: this code is very clunky, make it smarter
    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, y = ctx.saved_for_backward

        # If statements take care of high dimensional arrays
        dims_x = len(x.shape)
        dims_y = len(y.shape)

        if dims_y > 1:
            x_grad = np.matmul(grad, y.transpose(*range(dims_y - 2), -1, -2))
        else:
            x_grad = np.matmul(grad, y.T)
        if dims_x > 1:
            y_grad = np.matmul(x.transpose(*range(dims_x - 2), -1, -2), grad)
        else:
            y_grad = np.matmul(x.T, grad)

        # Adjust for broadcasting
        # Only works if added dim is in position 0, need to handle more?
        if dims_y > dims_x:
            x_grad = x_grad.sum(axis=0)
        if dims_x > dims_y:
            y_grad = y_grad.sum(axis=0)
            
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

class Expand(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, shape: tuple):
        ctx.save_for_backward(x.shape)
        return np.broadcast_to(x, shape)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        original_shape, = ctx.saved_for_backward
        n_dims_to_reduce = len(grad.shape) - len(original_shape)

        new_dims = tuple(np.arange(n_dims_to_reduce).tolist())
        expanded_dims = tuple(
            n_dims_to_reduce + i
            for i, (d1, d2) in enumerate(zip(original_shape, grad.shape[n_dims_to_reduce:]))
            if d1!=d2
        )

        out = np.sum(grad, axis=new_dims + expanded_dims, keepdims=False)
        if np.prod(original_shape) > 1:
            for dim in expanded_dims:
                out = np.expand_dims(out, axis=dim-n_dims_to_reduce)
 
        return out if np.prod(original_shape) > 1 else np.array([out])
register(Expand, "expand")

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, dims: tuple):
        ctx.save_for_backward(dims)
        return np.transpose(x, dims)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        dims, = ctx.saved_for_backward
        inverse_dims = np.argsort(dims)
        return np.transpose(grad, inverse_dims)
register(Permute, "permute")

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

# approximation of the error function https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erf.html
def approximate_erf(x: np.ndarray) -> np.ndarray:
    t = np.reciprocal(1 + 0.3275911 * np.abs(x))
    return np.sign(x) * (1 - ((((1.061405429 * t + -1.453152027) * t + 1.421413741) * t + -0.284496736) * t +0.254829592) * t * np.exp(-(np.square(x))))

class GELUoriginal(Function):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created.
    For information: OpenAI GPT's GELU is slightly different (and gives slightly different results).
    """
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        cdf = 0.5 * (1.0 + approximate_erf(x / 1.41421))
        ctx.save_for_backward(x, cdf)
        return x * cdf

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, cdf = ctx.saved_for_backward
        pdf = np.exp(-(np.log(np.sqrt(2 * np.pi)) + 0.5 * (x ** 2)))
        return grad * (cdf + x * pdf)
register(GELUoriginal, "gelu_original")

# Needed for backwards of GELU
def sech_squared(x: np.ndarray) -> np.ndarray:
    return np.square(2 / (np.exp(x) + np.exp(-x)))

class GELU(Function):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Uses the "tanh" approximation https://pytorch.org/docs/stable/generated/torch.nn.GELU.html.
    """
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        ctx.save_for_backward(x)
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        x, = ctx.saved_for_backward
        out = (
            0.5 * np.tanh(0.5 * (0.0713548 * np.power(x, 3) + 1.59577 * x)) 
            + (0.0535161 * np.power(x, 3) + 0.398942 * x) * sech_squared(0.0356774 * np.power(x, 3) + 0.797885 * x)
            + 0.5
        )
        return grad * out
register(GELU, "gelu")

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
    def backward(ctx: Context, grad: np.ndarray):
        out, = ctx.saved_for_backward
        return grad - np.exp(out)*grad.sum(axis=1).reshape((-1, 1))
register(LogSoftmax, "logsoftmax")

class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, dim: int):
        max_per_sample = np.max(x, axis=dim, keepdims=True)
        e_x = np.exp(x - max_per_sample)
        out = e_x / np.sum(e_x, axis=dim, keepdims=True)
        ctx.save_for_backward(out, dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        softmax_output, dim = ctx.saved_for_backward
        grad_out = softmax_output * (grad - np.sum(grad * softmax_output, axis=dim, keepdims=True))
        return grad_out
register(Softmax, "softmax")

