import numpy as np


class Optimizer:
    """
    Base Optimizer class.
    All optimizers should implement the update method.
    """
    def update(self, w, grad):
        raise NotImplementedError


# =========================
# 1. SGD (Vanilla)
# =========================
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, w, grad):
        return w - self.lr * grad


# =========================
# 2. SGD with Momentum
# =========================
class SGDMomentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None  # velocity

    def update(self, w, grad):
        if self.v is None:
            self.v = np.zeros_like(w)

        self.v = self.beta * self.v + grad
        return w - self.lr * self.v


# =========================
# 3. RMSProp
# =========================
class RMSProp(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = None  # squared gradient average

    def update(self, w, grad):
        if self.s is None:
            self.s = np.zeros_like(w)

        self.s = self.beta * self.s + (1 - self.beta) * (grad ** 2)
        return w - self.lr * grad / (np.sqrt(self.s) + self.eps)


# =========================
# 4. Adam Optimizer
# =========================
class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None  # first moment
        self.v = None  # second moment
        self.t = 0     # time step

    def update(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update weights
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =========================
# Simple Test / Demo
# =========================
if __name__ == "__main__":
    w = np.array([5.0])
    grad = np.array([2.0])

    optimizers = {
        "SGD": SGD(lr=0.1),
        "Momentum": SGDMomentum(lr=0.1),
        "RMSProp": RMSProp(lr=0.1),
        "Adam": Adam(lr=0.1),
    }

    for name, opt in optimizers.items():
        w_temp = w.copy()
        print(f"\n{name}")
        for step in range(5):
            w_temp = opt.update(w_temp, grad)
            print(f"Step {step + 1}: w = {w_temp}")
