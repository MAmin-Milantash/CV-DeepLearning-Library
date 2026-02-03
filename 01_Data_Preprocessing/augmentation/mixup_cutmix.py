"""
mixup_cutmix.py

Implements MixUp and CutMix augmentations.
"""

import numpy as np

def mixup(image1, image2, alpha=0.4):
    """MixUp two images with a given alpha."""
    lam = np.random.beta(alpha, alpha)
    mixed = lam * image1 + (1 - lam) * image2
    return mixed.astype(np.uint8)

def cutmix(image1, image2, alpha=1.0):
    """Cut a patch from image2 and place it into image1."""
    h, w = image1.shape[:2]
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # Random center
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    mixed = image1.copy()
    mixed[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    return mixed

# Example usage
if __name__ == "__main__":
    import cv2
    img1 = cv2.imread("sample1.jpg")
    img2 = cv2.imread("sample2.jpg")
    mixed = mixup(img1, img2)
    cv2.imwrite("mixed.jpg", mixed)
    cutmixed = cutmix(img1, img2)
    cv2.imwrite("cutmixed.jpg", cutmixed)
