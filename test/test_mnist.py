import gzip
import hashlib
from pathlib import Path

import numpy as np
import requests

from easygrad.tensor import Tensor


def fetch(url, mnist_path = Path("mnist")):
    if not mnist_path.exists():
        mnist_path.mkdir()
    file_path = mnist_path / hashlib.md5(url.encode("utf-8")).hexdigest()
    if file_path.exists():
        with open(file_path, "rb") as f:
            data = f.read()
    else:
        data = requests.get(url).content
        with open(file_path, "wb") as f:
            f.write(data)
    return data

def get_mnist():
    parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
    train_images = parse(fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28))
    train_labels = parse(fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))[8:]
    test_images = parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28))
    test_labels = parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"))[8:]
    return train_images, train_labels, test_images, test_labels

def layer_init_uniform(*shapes):
    ret = np.random.uniform(-1., 1., size=shapes) / np.sqrt(np.prod(shapes))
    return ret.astype(np.float32)

class TensorModel:
    def __init__(self):
        self.l1 = Tensor(layer_init_uniform(28*28, 256))
        self.l2 = Tensor(layer_init_uniform(256, 10))

    def __call__(self, x):
        assert(isinstance(x, Tensor)), "Only inputs of type Tensor allowed"
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

    def params(self):
        return [self.l1, self.l2]

def eval(model):
    x_test = Tensor(test_images)
    log_probs = model(x_test)
    preds = np.argmax(log_probs.data, axis=1)
    accuracy = np.equal(preds, test_labels).sum() / preds.size
    return accuracy


train_images, train_labels, test_images, test_labels = get_mnist()

def test_mnist():
    model = TensorModel()

    batch_size = 128
    steps = 500
    learning_rate = 0.001

    for step in range(steps):
        sample = np.random.randint(0, train_images.shape[0], size=batch_size)
        x_train = Tensor(train_images[sample])
        y_sample = train_labels[sample]
        y_train = np.zeros((batch_size, 10), dtype=np.float32)
        y_train[range(y_train.shape[0]), y_sample] = -10.0  # -10 because we have 10 classes
        y_train = Tensor(y_train)

        # Forward
        log_probs = model(x_train)

        # NLL loss
        nll = log_probs.mul(y_train).mean()

        # Backward
        for param in model.params():
            param.grad = None
        nll.backward()

        # Update
        for param in model.params():
            param.data -= learning_rate * param.grad

        # Test
        # if step % 20 == 0:
        #     preds = np.argmax(log_probs.data, axis=1)
        #     accuracy = np.equal(preds, y_sample).sum() / preds.size
        #     test_accuracy = eval(model)
        #     print(f"Step: {step:03d}, loss: {nll.data.item():.5f}, accuracy: {accuracy:.4f}, test accuracy: {test_accuracy:.4f}")

    test_accuracy = eval(model)
    assert test_accuracy > 0.94
