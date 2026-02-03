"""
rotations.py

Implements image rotation by random angles for augmentation.
"""

import cv2
import numpy as np

def rotate_image(image, angle):
    """Rotate the image by a given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def random_rotation(image, max_angle=30):
    """Rotate the image by a random angle between -max_angle and max_angle."""
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate_image(image, angle)

# Example usage
if __name__ == "__main__":
    img = cv2.imread("sample.jpg")
    rotated = random_rotation(img)
    cv2.imwrite("rotated.jpg", rotated)
