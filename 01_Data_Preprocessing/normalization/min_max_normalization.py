"""
Min-Max Normalization (from scratch)

Transforms input data to a fixed range [0, 1].

Useful for:
- Pixel values in images
- Any feature scaling before feeding into neural networks
"""

import numpy as np

def min_max_normalize(x, feature_range=(0, 1)):
    """
    Normalize input array x to the given range.
    
    Args:
        x (np.ndarray): Input array (image or feature map)
        feature_range (tuple): Desired output range (min, max)

    Returns:
        np.ndarray: Normalized array
    """
    x_min = np.min(x)
    x_max = np.max(x)
    scale_min, scale_max = feature_range

    # Avoid division by zero
    if x_max - x_min == 0:
        return np.zeros_like(x) + scale_min

    normalized = (x - x_min) / (x_max - x_min)  # Scale to [0, 1]
    normalized = normalized * (scale_max - scale_min) + scale_min  # Scale to feature_range
    return normalized

# -------------------------------
# Example usage:
if __name__ == "__main__":
    # Random image array (0-255)
    image = np.random.randint(0, 256, size=(5, 5, 3), dtype=np.uint8)
    print("Original Image:\n", image)

    norm_image = min_max_normalize(image)
    print("\nNormalized Image [0,1]:\n", norm_image)

    # Normalize to [-1, 1]
    norm_image2 = min_max_normalize(image, feature_range=(-1, 1))
    print("\nNormalized Image [-1,1]:\n", norm_image2)
