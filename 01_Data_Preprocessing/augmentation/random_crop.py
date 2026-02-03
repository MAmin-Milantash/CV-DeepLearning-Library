"""
random_crop.py

Performs random cropping of images.
"""

import cv2
import numpy as np

def random_crop(image, crop_size):
    """Randomly crop a patch from the image."""
    h, w = image.shape[:2]
    ch, cw = crop_size
    if h < ch or w < cw:
        raise ValueError("Crop size must be smaller than image size.")
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return image[y:y+ch, x:x+cw]

# Example usage
if __name__ == "__main__":
    img = cv2.imread("sample.jpg")
    cropped = random_crop(img, (200, 200))
    cv2.imwrite("cropped.jpg", cropped)
