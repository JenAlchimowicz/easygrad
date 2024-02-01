import numpy as np


class SGD:
    def __init__(self, params, learning_rate):
        self.params = params
        self.lr = learning_rate

    def step(self):
        for param in self.params:
            param.data -= param.grad * self.lr


class RMSprop:
    def __init__(self, params, learning_rate, decay=0.999, eps=1e-8):
        self.params = params
        self.lr = learning_rate
        self.decay = decay
        self.eps = eps
        self.s = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.s[i] = self.decay * self.s[i] + (1 - self.decay) * np.square(param.grad)
            param.data -= self.lr * param.grad / np.sqrt(self.s[i] + self.eps)


class Adam:
    def __init__(self, params, learning_rate, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.v = [np.zeros_like(param.data) for param in self.params]
        self.s = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            self.v[i] = self.b1 * self.v[i] + (1 - self.b1) * param.grad
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * np.square(param.grad)

            # Correction
            self.v[i] = self.v[i] / np.sqrt(1 - np.pow(self.b1, self.t))
            self.s[i] = self.s[i] / np.sqrt(1 - np.pow(self.b1, self.t))

            # Update
            param.data -= self.lr * self.v[i] / np.sqrt(self.s[i] + self.eps)
