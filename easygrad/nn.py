import numpy as np

from easygrad.tensor import Tensor
from easygrad.init import xavier_uniform


class Embedding:
    def __init__(self, size: int, embed_dim: int):
        self.weights = Tensor(xavier_uniform(size, embed_dim))
        self.vocab_counter = np.arange(size).reshape(1, 1, size)

    def __call__(self, idx: np.ndarray):
        # TODO: add a check if shapes are ok
        out = Tensor(self.vocab_counter == np.expand_dims(idx, 2))
        return out.dot(self.weights)

class LayerNorm:
    def __init__(self, normalized_shape, eps: float = 1e-5):
        assert(isinstance(normalized_shape, int)), f"Only 1d LayerNorm supported, normalized shape should be an int, got {normalized_shape}"
        self.gamma = Tensor(np.ones(normalized_shape))
        self.beta = Tensor(np.zeros(normalized_shape))
        self.eps = eps

    def __call__(self, x: Tensor):
        last_dim = len(x.shape) - 1
        mean = x.mean(axis=last_dim, keepdims=True)
        var = x.sub(mean.expand(shape=x.shape)).square().mean(axis=last_dim, keepdims=True)
        eps = Tensor(np.ones(var.shape) * self.eps)
        std = (var.add(eps)).sqrt()
        y = (x.sub(mean.expand(shape=x.shape))).div(std.expand(shape=x.shape))
        out = y.mul(self.gamma.expand(shape=x.shape)).add(self.beta.expand(shape=x.shape))
        return out
