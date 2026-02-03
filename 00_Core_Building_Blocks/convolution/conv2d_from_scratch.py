import numpy as np
from utils import pad_2d, compute_output_size

def conv2d(input_matrix, kernel, stride=1, padding=0):
    input_matrix = pad_2d(input_matrix, padding)

    h_out = compute_output_size(
        input_matrix.shape[0], kernel.shape[0], stride, 0
    )
    w_out = compute_output_size(
        input_matrix.shape[1], kernel.shape[1], stride, 0
    )

    output = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            region = input_matrix[
                h_start:h_start + kernel.shape[0],
                w_start:w_start + kernel.shape[1]
            ]
            output[i, j] = np.sum(region * kernel)

    return output


if __name__ == "__main__":
    x = np.random.rand(5, 5)
    kernel = np.array([[1, 0], [0, -1]])

    y = conv2d(x, kernel, stride=1, padding=1)
    print("Conv2D Output:\n", y)
