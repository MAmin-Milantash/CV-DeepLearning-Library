import numpy as np

class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, d_out):
        return d_out * self.out * (1 - self.out)

# Example usage:
# sig = Sigmoid()
# y = sig.forward(np.array([-1.0, 0.0, 1.0]))
# grad = sig.backward(np.ones_like(y))
