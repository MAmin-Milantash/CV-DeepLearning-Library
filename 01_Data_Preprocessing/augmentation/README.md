# Data Augmentation

This folder contains commonly used image augmentation techniques applied before training deep learning models.  
The main goal is to:
- Prevent overfitting
- Make the model robust to real-world variations
- Improve generalization

## Folder Structure
01_Data_Preprocessing/
└── augmentation/
├── flips.py
├── rotations.py
├── color_jitter.py
├── random_crop.py
├── mixup_cutmix.py
└── README.md

## File Descriptions

**flips.py**  
- Implements horizontal and vertical flips for images.  
- **Purpose:** Increases dataset variety without changing the semantic content.

**rotations.py**  
- Implements image rotation by random angles.  
- **Purpose:** Makes the model invariant to different orientations.

**color_jitter.py**  
- Changes brightness, contrast, saturation, and hue.  
- **Purpose:** Makes the model robust to lighting and color variations in real-world images.

**random_crop.py**  
- Performs random cropping of images.  
- **Purpose:** Forces the model to learn from different parts of the image and prevents overfitting.

**mixup_cutmix.py**  
- Implements MixUp (linear combination of two images) and CutMix (replace a patch of one image with another).  
- **Purpose:** Improves generalization and diversity in training.

## Recommended Workflow

1. Apply **flips**  
2. Apply **rotations**  
3. Apply **color jitter**  
4. Apply **random crop**  
5. Apply **MixUp / CutMix**

> Interview tip: Always apply augmentation **only to training data**, never to validation or test sets.

