import numpy as np

def zero_init(shape):
    """Zero initialization"""
    return np.zeros(shape)

def random_normal_init(shape, mean=0.0, std=0.01):
    """Random normal initialization"""
    return np.random.normal(mean, std, shape)

def random_uniform_init(shape, low=-0.01, high=0.01):
    """Random uniform initialization"""
    return np.random.uniform(low, high, shape)

def xavier_init(shape):
    """Xavier/Glorot initialization for Sigmoid/Tanh"""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def he_init(shape):
    """He initialization for ReLU"""
    fan_in = shape[0]
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, shape)

# Example usage
if __name__ == "__main__":
    w_zero = zero_init((3, 3))
    w_normal = random_normal_init((3, 3))
    w_uniform = random_uniform_init((3, 3))
    w_xavier = xavier_init((3, 3))
    w_he = he_init((3, 3))

    print("Zero Init:\n", w_zero)
    print("Normal Init:\n", w_normal)
    print("Uniform Init:\n", w_uniform)
    print("Xavier Init:\n", w_xavier)
    print("He Init:\n", w_he)
