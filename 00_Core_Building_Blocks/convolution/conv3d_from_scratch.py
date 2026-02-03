import numpy as np
from utils import pad_3d, compute_output_size

def conv3d(volume, kernel, stride=1, padding=0):
    volume = pad_3d(volume, padding)

    d_out = compute_output_size(volume.shape[0], kernel.shape[0], stride, 0)
    h_out = compute_output_size(volume.shape[1], kernel.shape[1], stride, 0)
    w_out = compute_output_size(volume.shape[2], kernel.shape[2], stride, 0)

    output = np.zeros((d_out, h_out, w_out))

    for d in range(d_out):
        for i in range(h_out):
            for j in range(w_out):
                d_start = d * stride
                i_start = i * stride
                j_start = j * stride

                region = volume[
                    d_start:d_start + kernel.shape[0],
                    i_start:i_start + kernel.shape[1],
                    j_start:j_start + kernel.shape[2]
                ]

                output[d, i, j] = np.sum(region * kernel)

    return output


if __name__ == "__main__":
    x = np.random.rand(4, 4, 4)
    kernel = np.ones((2, 2, 2))

    y = conv3d(x, kernel)
    print("Conv3D Output shape:", y.shape)
