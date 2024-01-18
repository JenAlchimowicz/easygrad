from functools import partialmethod

import numpy as np


class Tensor:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            assert(False), "Only np arrays as inputs are allowed"

        self.data = data
        self.grad = None

        # For autograd
        self._ctx = None

    def backward(self, allow_fill=True):
        if self._ctx is None:  # Noting to backpropagate on
            return 

        if self.grad is None and allow_fill:
            assert(self.data.size == 1), "Can only start backprop on scalar values such as a loss"
            self.grad = np.array([1])

        grads = self._ctx.backward(self._ctx, self.grad)  # Grads of the 2 matrices that created self
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            # if g is None:
            #    continue
            # assert(g.shape == t.data.shape), f"Gradient must be the same shape as data, grad shape: {g.shape}, data shape: {t.data.shape}"
            t.grad = g
            t.backward(False)

    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum().mul(div)

    def __repr__(self):
        return f"Tensor of size {self.data.shape}, data {self.data} and grad {self.grad}"
      

class Function:
  def __init__(self, *tensors) -> None:
    self.parents = tensors
    self.saved_tensors = []

  def apply(self, arg, *x): #tensor, #op(fixed), other tensors (optional)
    if type(arg) == Tensor:
      op = self
      x = [arg]+list(x)
    else:
      op = arg
      x = [self]+list(x)
    ctx = op(*x)
    ret = Tensor(op.forward(ctx, *[t.data for t in x]))
    ret._ctx = ctx
    return ret

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

def register(name, fxn):
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

# **** implement a few functions ****

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x*y

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return y*grad_output, x*grad_output
register('mul', Mul)

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    def backward(ctx, grad):
        return grad, grad
register("add", Add)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.array(input.sum())

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register('sum', Sum)


class Dot(Function):
  @staticmethod
  def forward(ctx, x, weight):
    ctx.save_for_backward(x, weight)
    return x.dot(weight)

  @staticmethod
  def backward(ctx, grad_output):
    x, w = ctx.saved_tensors
    grad_x = grad_output.dot((w.T))
    grad_w = x.T.dot(grad_output)
    return grad_x, grad_w
register('dot', Dot)


class ReLU(Function):
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return np.maximum(x, 0)

  @staticmethod
  def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    grad_output[x<0] = 0
    return grad_output
register('relu', ReLU)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, x):
        e_x = np.exp(x - x.max(axis=1, keepdims=True))
        log_softmax = np.log(e_x / e_x.sum(axis=1, keepdims=True))
        ctx.save_for_backward(log_softmax)
        return log_softmax

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)    


class Conv2D(Function):
  @staticmethod
  def inner_forward(x, w):
    cout,cin,H,W = w.shape
    ret = np.zeros((x.shape[0], cout, x.shape[2]-(H-1), x.shape[3]-(W-1)), dtype=w.dtype)
    for j in range(H):
      for i in range(W):
        tw = w[:, :, j, i]
        for Y in range(ret.shape[2]):
          for X in range(ret.shape[3]):
            ret[:, :, Y, X] += x[:, :, Y+j, X+i].dot(tw.T)
    return ret

  @staticmethod
  def inner_backward(grad_output, x, w):
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    cout,cin,H,W = w.shape
    for j in range(H):
      for i in range(W):
        tw = w[:, :, j, i]
        for Y in range(grad_output.shape[2]):
          for X in range(grad_output.shape[3]):
            gg = grad_output[:, :, Y, X]
            tx = x[:, :, Y+j, X+i]
            dx[:, :, Y+j, X+i] += gg.dot(tw)
            dw[:, :, j, i] += gg.T.dot(tx)
    return dx, dw

  @staticmethod
  def forward(ctx, x, w):
    ctx.save_for_backward(x, w)
    return Conv2D.inner_forward(x, w)

  @staticmethod
  def backward(ctx, grad_output):
    return Conv2D.inner_backward(grad_output, *ctx.saved_tensors)
register('conv2d', Conv2D)