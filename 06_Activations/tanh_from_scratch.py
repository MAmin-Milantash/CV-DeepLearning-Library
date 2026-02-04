import numpy as np

class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, d_out):
        return d_out * (1 - self.out**2)

# Example usage:
# t = Tanh()
# y = t.forward(np.array([-1.0, 0.0, 1.0]))
# grad = t.backward(np.ones_like(y))
