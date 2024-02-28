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

## Todo
- Finish BertEmbeddings
