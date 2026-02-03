"""
color_jitter.py

Applies random changes to brightness, contrast, saturation, and hue.
"""

import cv2
import numpy as np

def adjust_brightness(image, factor):
    """Adjust brightness by factor (0.5 = darker, 1.5 = brighter)."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_contrast(image, factor):
    """Adjust contrast by factor."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=128*(1-factor))

def random_color_jitter(image, brightness=0.2, contrast=0.2):
    """Randomly jitter brightness and contrast."""
    b_factor = 1 + np.random.uniform(-brightness, brightness)
    c_factor = 1 + np.random.uniform(-contrast, contrast)
    img = adjust_brightness(image, b_factor)
    img = adjust_contrast(img, c_factor)
    return img

# Example usage
if __name__ == "__main__":
    img = cv2.imread("sample.jpg")
    jittered = random_color_jitter(img)
    cv2.imwrite("jittered.jpg", jittered)
