import numpy as np
from scipy import stats
import torch

from easygrad.init import xavier_uniform, xavier_normal


def helper_test_op(easygrad_fn, torch_fn, n_runs=200, min_pass_rate=0.9):
    passes, fails = 0, 0

    for _ in range(n_runs):
        shapes = np.random.randint(1, 100, size=2)

        np_arr = easygrad_fn(*shapes).flatten()
        w = torch.empty(*shapes)
        torch_arr = torch_fn(w).detach().numpy().flatten()

        _, p_value = stats.kstest(np_arr, torch_arr)
        if p_value > 0.05:
            passes += 1
        else:
            fails += 1
    
    pass_rate = passes / (passes + fails)
    assert(pass_rate >= min_pass_rate)


class TestInits:
    def test_xavier_uniform(self):
        helper_test_op(xavier_uniform, torch.nn.init.xavier_uniform_)
    def test_xavier_normal(self):
        helper_test_op(xavier_normal, torch.nn.init.xavier_normal_)
