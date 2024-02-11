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
