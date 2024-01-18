import requests
import gzip
import os
import hashlib
import numpy as np

from tensor import Tensor
from optim import SGD, Adam

# Get data
def fetch(url):
  fp = os.path.join("mnist", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]


# Create model
def layer_init(m, h):
  ret = np.random.uniform(-1.0, 1.0, size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

class MyModel:
  def __init__(self):
    self.l1 = Tensor(layer_init(784, 128))
    self.l2 = Tensor(layer_init(128, 10))
  
  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


# Optimizer



# Train loop
model = MyModel()
optim = SGD([model.l1, model.l2], lr=0.01)
batch_size = 128
lr = 0.01

losses, accuracies = [], []

for i in range(1001):
    sample = np.random.randint(0, X_train.shape[0], batch_size)
    x = Tensor(X_train[sample].reshape(-1, 28*28))
    Y = Y_train[sample]
    y = np.zeros((128, 10), np.float32)
    y[range(y.shape[0]), Y_train[sample]] = -1
    y = Tensor(y)

    # Forward
    outs = model.forward(x)
    loss = outs.mul(y).mean()

    # Backprop
    loss.backward()
    optim.step()

    # Metrics
    cat = np.argmax(outs.data, axis=1)
    accuracy = (cat == Y).mean()

    losses.append(loss)
    accuracies.append(accuracy)

    if i % 50 == 0:
        print(f"Batch: {i}, accuracy: {accuracy.item():.4f}, loss: {loss.data.item():.4f}")


# evaluate
def numpy_eval():
  Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
  Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
  return (Y_test == Y_test_preds).mean()

accuracy = numpy_eval()
print("test set accuracy is %f" % accuracy)
assert accuracy > 0.95
