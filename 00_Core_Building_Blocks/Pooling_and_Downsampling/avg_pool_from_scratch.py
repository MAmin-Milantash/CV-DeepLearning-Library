import numpy as np

def avg_pool2d(input, kernel_size=2, stride=2):
    H, W = input.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride

            window = input[h_start:h_start+kernel_size,
                            w_start:w_start+kernel_size]

            output[i, j] = np.mean(window)

    return output


if __name__ == "__main__":
    x = np.array([
        [1, 3, 2, 1],
        [4, 6, 5, 2],
        [1, 2, 0, 1],
        [3, 1, 2, 4]
    ])

    print(avg_pool2d(x))
