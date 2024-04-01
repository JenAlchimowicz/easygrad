# easygrad
Automatic gradient engine implemented in numpy, inspired by [micrograd](https://github.com/karpathy/micrograd) and [tinygrad](https://github.com/tinygrad/tinygrad).

More complex than micrograd, operates on Tensors directly. <br>
Less complex than tinygrad, without all the accelerator support.

## Example
The engine can support a lot of basic operations such as add, mul, matmul as well as some tensor operations such as reshape and expand. Full list of ops in [easygrad/tensor.py](easygrad/tensor.py).

### Quick example comparing to PyTorch
```python
from easygrad.tensor import Tensor
import numpy as np

a = np.random.rand(1, 3).astype(np.float32)
b = np.random.rand(3, 2).astype(np.float32)
a = Tensor(a)
b = Tensor(b)

c = a.dot(b).relu()
d = c.sum()
d.backward()

print(a.grad)  # dd/da
print(b.grad)  # dd/db
```

The same thing but in PyTorch
```python
import torch
import numpy as np

a = np.random.rand(1, 3).astype(np.float32)
b = np.random.rand(3, 2).astype(np.float32)
a = torch.tensor(a, requires_grad=True)
b = torch.tensor(b, requires_grad=True)

c = a.matmul(b).relu()
d = c.sum()
d.backward()

print(a.grad.numpy())  # dd/da
print(b.grad.numpy())  # dd/db
```

### Neural Networks
We have a few NN utilities such as optimizers or weight initializations. All tested for consistency with PyTorch. For more complex examples (e.g. BERT) see [examples](examples) or [test on mnist](test/test_mnist.py).
```python
import numpy as np
from easygrad import Tensor
from easygrad.init import xavier_uniform
from easygrad.optim import Adam

class EasyNet:
    def __init__(self):
        self.l1 = Tensor(xavier_uniform(4, 8))
        self.l2 = Tensor(xavier_uniform(8, 1))

    def __call__(self, x: Tensor):
        return x.dot(self.l1).relu().dot(self.l2)

model = EasyNet()
optim = Adam([model.l1, model.l2], lr=3e-2)

x = Tensor(np.random.rand(32, 4))
y = Tensor(np.random.rand(32, 1))

for i in range(10):
    out = model(x)
    loss = (y - out).square().mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"{i}, loss: {loss.data}")
```

## Todo
- Fix backward for bert and add example for full training
- Add a more modern transformer to examples
