def compute_receptive_field(layers):
    """
    layers: list of dicts
    Each layer: {kernel, stride}
    """
    rf = 1
    jump = 1

    for layer in layers:
        k = layer["kernel"]
        s = layer["stride"]

        rf = rf + (k - 1) * jump
        jump *= s

    return rf


if __name__ == "__main__":
    cnn = [
        {"kernel": 3, "stride": 1},
        {"kernel": 3, "stride": 2},
        {"kernel": 3, "stride": 1},
    ]

    print("Receptive Field:", compute_receptive_field(cnn))
