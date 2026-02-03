"""
Standardization / Z-score Normalization (from scratch)

Transforms input data to have zero mean and unit variance.

Useful for:
- Features with different scales
- Helps faster convergence in neural networks
- Often used with CNNs and fully connected layers
"""

import numpy as np

def standardize(x):
    """
    Standardize input array x (zero mean, unit variance)
    
    Args:
        x (np.ndarray): Input array (image or feature map)
        
    Returns:
        np.ndarray: Standardized array
    """
    mean = np.mean(x)
    std = np.std(x)

    # Avoid division by zero
    if std == 0:
        return np.zeros_like(x)
    
    standardized = (x - mean) / std
    return standardized

# -------------------------------
# Example usage
if __name__ == "__main__":
    image = np.random.randint(0, 256, size=(5,5,3), dtype=np.uint8)
    print("Original Image:\n", image)

    standardized_image = standardize(image)
    print("\nStandardized Image:\n", standardized_image)
