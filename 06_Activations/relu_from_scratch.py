import numpy as np

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask

# Example usage:
# r = ReLU()
# y = r.forward(np.array([-1.0, 0.0, 1.0]))
# grad = r.backward(np.ones_like(y))
