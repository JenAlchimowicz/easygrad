import numpy as np


def xavier_uniform(*shapes):
    out = np.random.uniform(-1, 1, size=shapes) * np.sqrt(6 / (shapes[0] + np.prod(shapes[1:])))
    return out.astype(np.float32)

def xavier_normal(*shapes):
    std = np.sqrt(2 / (shapes[0] + np.prod(shapes[1:])))
    out = np.random.normal(0, std, size=shapes)
    return out.astype(np.float32)
