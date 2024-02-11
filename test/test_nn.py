import numpy as np
import torch

from easygrad.nn import Embedding
from easygrad.tensor import Tensor


def test_embedding():
    embed_size = 100
    embed_dim = 5
    weights_np = np.random.rand(embed_size, embed_dim).astype(np.float32)
    weights_pt = torch.from_numpy(weights_np)

    easy_embed = Embedding(embed_size, embed_dim)
    easy_embed.weights = Tensor(weights_np)
    torch_embed = torch.nn.Embedding.from_pretrained(weights_pt, freeze=False)

    # Forward
    idx_np = np.random.randint(0, embed_size, size=(4,10))
    out_easy = easy_embed(idx_np)

    idx_pt = torch.from_numpy(idx_np)
    out_torch = torch_embed(idx_pt)
    np.testing.assert_allclose(out_easy.data, out_torch.detach().numpy())

    # Backward
    loss_easy = out_easy.mean()
    loss_easy.backward()

    loss_torch = out_torch.mean()
    loss_torch.backward()
    np.testing.assert_allclose(easy_embed.weights.grad, torch_embed.weight.grad.detach().numpy())
