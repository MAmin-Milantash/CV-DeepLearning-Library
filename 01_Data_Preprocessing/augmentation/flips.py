"""
flips.py

Implements horizontal and vertical flips for image augmentation.
"""

import numpy as np
import cv2

def horizontal_flip(image):
    """Flip the image horizontally."""
    return cv2.flip(image, 1)

def vertical_flip(image):
    """Flip the image vertically."""
    return cv2.flip(image, 0)

# Example usage
if __name__ == "__main__":
    img = cv2.imread("sample.jpg")
    h_flip = horizontal_flip(img)
    v_flip = vertical_flip(img)
    cv2.imwrite("h_flip.jpg", h_flip)
    cv2.imwrite("v_flip.jpg", v_flip)
