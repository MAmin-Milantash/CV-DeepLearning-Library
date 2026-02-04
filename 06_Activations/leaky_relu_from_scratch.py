import numpy as np

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, d_out):
        dx = np.ones_like(self.x)
        dx[self.x < 0] = self.alpha
        return d_out * dx

# Example usage:
# lrelu = LeakyReLU(alpha=0.01)
# y = lrelu.forward(np.array([-1.0, 0.0, 1.0]))
# grad = lrelu.backward(np.ones_like(y))
