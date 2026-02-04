import math

# =====================================
# Base Scheduler Class
# =====================================
class LRScheduler:
    """
    Base class for learning rate schedulers.
    """
    def __init__(self, lr_init=0.01):
        self.lr_init = lr_init
        self.current_step = 0

    def step(self):
        """
        Update learning rate and return it.
        """
        self.current_step += 1
        return self.get_lr()

    def get_lr(self):
        """
        Compute learning rate. Override in subclasses.
        """
        return self.lr_init


# =====================================
# Step Decay Scheduler
# =====================================
class StepDecay(LRScheduler):
    """
    Reduces LR by factor every `step_size` steps.
    """
    def __init__(self, lr_init=0.01, step_size=10, gamma=0.1):
        super().__init__(lr_init)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        factor = self.gamma ** (self.current_step // self.step_size)
        return self.lr_init * factor


# =====================================
# Exponential Decay Scheduler
# =====================================
class ExponentialDecay(LRScheduler):
    """
    Reduces LR exponentially every step.
    """
    def __init__(self, lr_init=0.01, gamma=0.95):
        super().__init__(lr_init)
        self.gamma = gamma

    def get_lr(self):
        return self.lr_init * (self.gamma ** self.current_step)


# =====================================
# Cosine Annealing Scheduler
# =====================================
class CosineAnnealing(LRScheduler):
    """
    Cosine annealing schedule for cyclical learning rates.
    """
    def __init__(self, lr_init=0.01, T_max=50, lr_min=0.0):
        super().__init__(lr_init)
        self.T_max = T_max
        self.lr_min = lr_min

    def get_lr(self):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.current_step % self.T_max) / self.T_max))
        lr = self.lr_min + (self.lr_init - self.lr_min) * cosine_decay
        return lr


# =====================================
# Demo / Example Usage
# =====================================
if __name__ == "__main__":
    schedulers = {
        "step_decay": StepDecay(lr_init=0.1, step_size=5, gamma=0.5),
        "exponential_decay": ExponentialDecay(lr_init=0.1, gamma=0.9),
        "cosine_annealing": CosineAnnealing(lr_init=0.1, T_max=10, lr_min=0.01)
    }

    steps = 20

    for name, scheduler in schedulers.items():
        print(f"\n{name.upper()}:")
        for step in range(steps):
            lr = scheduler.step()
            print(f"Step {step + 1}: LR = {lr:.5f}")
