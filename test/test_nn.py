import numpy as np
import torch

from easygrad.nn import Embedding, LayerNorm
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


def test_layer_norm():
    normalized_shape = 4
    x_torch = torch.randn(2, 2, normalized_shape)
    x_easy = Tensor(x_torch.detach().numpy())

    ### Default intialization (gamma=1, beta=0)
    ln_torch = torch.nn.LayerNorm(normalized_shape)
    ln_easy = LayerNorm(normalized_shape)
    
    # Forward
    out_torch = ln_torch(x_torch)
    out_easy = ln_easy(x_easy)
    np.testing.assert_allclose(out_easy.data, out_torch.detach().numpy(), atol=1e-6)

    # Backward
    loss_torch = out_torch.mean()
    loss_torch.backward()
    loss_easy = out_easy.mean()
    loss_easy.backward()
    np.testing.assert_allclose(ln_easy.gamma.grad, ln_torch.weight.grad.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(ln_easy.beta.grad, ln_torch.bias.grad.detach().numpy(), atol=1e-6)

    ### Custom random initialization
    gamma = torch.rand(normalized_shape)
    beta = torch.rand(normalized_shape)

    ln_easy = LayerNorm(normalized_shape)
    ln_easy.gamma.data = gamma.detach().numpy()
    ln_easy.beta.data = beta.detach().numpy()
    ln_torch = torch.nn.LayerNorm(normalized_shape)
    ln_torch.weight = torch.nn.Parameter(gamma)
    ln_torch.bias = torch.nn.Parameter(beta)

    # Forward
    out_easy = ln_easy(x_easy)
    out_torch = ln_torch(x_torch)
    np.testing.assert_allclose(out_easy.data, out_torch.detach().numpy(), atol=1e-6)

    # Backward
    loss_easy = out_easy.mean()
    loss_easy.backward()
    loss_torch = out_torch.mean()
    loss_torch.backward()
    np.testing.assert_allclose(ln_easy.gamma.grad, ln_torch.weight.grad.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(ln_easy.beta.grad, ln_torch.bias.grad.detach().numpy(), atol=1e-6)
