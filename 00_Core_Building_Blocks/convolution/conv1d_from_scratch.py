import numpy as np
from utils import pad_1d, compute_output_size

def conv1d(input_signal, kernel, stride=1, padding=0):
    input_signal = pad_1d(input_signal, padding)

    output_size = compute_output_size(
        len(input_signal), len(kernel), stride, 0
    )

    output = np.zeros(output_size)

    for i in range(output_size):
        start = i * stride
        end = start + len(kernel)
        output[i] = np.sum(input_signal[start:end] * kernel)

    return output


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    kernel = np.array([1, 0, -1])

    y = conv1d(x, kernel, stride=1, padding=1)
    print("Conv1D Output:", y)
