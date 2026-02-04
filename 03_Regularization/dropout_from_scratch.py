import numpy as np

def dropout_forward(x, drop_prob=0.5, training=True):
    """
    x: input activations (numpy array)
    drop_prob: probability of dropping a neuron
    training: whether in training mode
    """
    if not training or drop_prob == 0.0:
        return x

    keep_prob = 1.0 - drop_prob

    # binary mask
    mask = (np.random.rand(*x.shape) < keep_prob).astype(np.float32)

    # inverted dropout
    out = (x * mask) / keep_prob
    return out


if __name__ == "__main__":
    x = np.ones((3, 5))
    out = dropout_forward(x, drop_prob=0.5, training=True)
    print(out)
    
""" Inverted Dropout eliminates the need for scaling in inference. """
