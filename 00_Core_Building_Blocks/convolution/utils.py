import numpy as np

def compute_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def pad_1d(x, padding):
    if padding == 0:
        return x
    return np.pad(x, (padding, padding), mode='constant')


def pad_2d(x, padding):
    if padding == 0:
        return x
    return np.pad(x, ((padding, padding), (padding, padding)), mode='constant')


def pad_3d(x, padding):
    if padding == 0:
        return x
    return np.pad(x, ((padding, padding),
                      (padding, padding),
                      (padding, padding)), mode='constant')
