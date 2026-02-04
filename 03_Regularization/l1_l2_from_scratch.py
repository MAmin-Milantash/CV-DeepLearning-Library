import numpy as np

def l1_regularization(weights, lambda_l1):
    return lambda_l1 * np.sum(np.abs(weights))


def l2_regularization(weights, lambda_l2):
    return lambda_l2 * np.sum(weights ** 2)


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


if __name__ == "__main__":
    weights = np.random.randn(10)
    y_pred = np.array([2.5, 0.0, 2.1])
    y_true = np.array([3.0, -0.5, 2.0])

    loss = mse_loss(y_pred, y_true)
    loss += l1_regularization(weights, 0.01)
    loss += l2_regularization(weights, 0.01)

    print("Total loss:", loss)

"""
    L1 → sparse

    L2 → smooth
"""