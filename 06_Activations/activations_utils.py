import numpy as np
import matplotlib.pyplot as plt

def plot_activation(func, x_range=(-5,5), num=100):
    x = np.linspace(x_range[0], x_range[1], num)
    y = func.forward(x)
    plt.plot(x, y, label=func.__class__.__name__)
    plt.title(f'{func.__class__.__name__} Activation')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_gradients(func, x_range=(-5,5), num=100):
    x = np.linspace(x_range[0], x_range[1], num)
    func.forward(x)
    grad = func.backward(np.ones_like(x))
    plt.plot(x, grad, label=f'{func.__class__.__name__} Gradient')
    plt.title(f'{func.__class__.__name__} Gradient')
    plt.grid(True)
    plt.legend()
    plt.show()
