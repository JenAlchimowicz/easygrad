import numpy as np

from easygrad.tensor import Tensor
from easygrad.init import xavier_uniform


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = Tensor(xavier_uniform(in_features, out_features))
        self.bias = Tensor(np.zeros((1, out_features))) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x.dot(self.weight)
        if self.bias is not None:
            out = out.add(self.bias.expand(shape=out.shape))
        return out
    
class Embedding:
    def __init__(self, size: int, embed_dim: int):
        self.weight = Tensor(xavier_uniform(size, embed_dim))
        self.vocab_counter = np.arange(size).reshape(1, 1, size)

    def __call__(self, idx: np.ndarray):
        # TODO: add a check if shapes are ok
        out = Tensor(self.vocab_counter == np.expand_dims(idx, 2))
        return out.dot(self.weight)

class LayerNorm:
    def __init__(self, normalized_shape, eps: float = 1e-5):
        assert(isinstance(normalized_shape, int)), f"Only 1d LayerNorm supported, normalized shape should be an int, got {normalized_shape}"
        self.weight = Tensor(np.ones(normalized_shape))
        self.bias = Tensor(np.zeros(normalized_shape))
        self.eps = eps

    def __call__(self, x: Tensor):
        last_dim = len(x.shape) - 1
        mean = x.mean(axis=last_dim, keepdims=True)
        var = x.sub(mean.expand(shape=x.shape)).square().mean(axis=last_dim, keepdims=True)
        eps = Tensor(np.ones(var.shape) * self.eps)
        std = (var.add(eps)).sqrt()
        y = (x.sub(mean.expand(shape=x.shape))).div(std.expand(shape=x.shape))
        out = y.mul(self.weight.expand(shape=x.shape)).add(self.bias.expand(shape=x.shape))
        return out

class Dropout:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: Tensor):
        if not x.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.p) * (1/(1.0-self.p))
        mask = Tensor(mask.astype(np.float32))
        return x.mul(mask)
