import numpy as np

class BatchNorm:
    def __init__(self, dim, epsilon=1e-5, momentum=0.9):
        self.dim = dim
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones(dim)  # Scale
        self.beta = np.zeros(dim)  # Shift
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def forward(self, x, training=True):
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            # Use running mean/var at inference
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        out = self.gamma * x_norm + self.beta
        return out
