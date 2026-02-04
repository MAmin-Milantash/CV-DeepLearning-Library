# 📁 08_Transfer_Learning

## 🧭 Purpose

This folder covers **Transfer Learning (TL) techniques**, which leverage **pre-trained models** to solve new tasks efficiently.

Key aspects include:

* Using pre-trained networks as **feature extractors**.
* **Fine-tuning** pre-trained weights for task-specific learning.
* Loading and adapting popular pre-trained models.
* Utility functions for evaluation and dataset adaptation.

Transfer Learning is **critical for modern deep learning**, especially when labeled data is limited.

---

## 🧭 Why Transfer Learning Matters

* **Reduces training time**: pre-trained weights already capture rich representations.
* **Improves performance**: helps achieve high accuracy even with small datasets.
* **Enables practical applications**: can use state-of-the-art models without training from scratch.
* **Supports domain adaptation**: e.g., medical imaging, satellite imagery, NLP tasks.

It logically follows **Architectural Blocks**, as TL uses the same blocks in pre-trained networks but focuses on **reuse and adaptation**.

---

## 🪜 Recommended Learning Order

1️⃣ **Pre-trained models overview** – understand architectures like ResNet, VGG, EfficientNet.
2️⃣ **Feature extraction** – freeze layers and use the network as a fixed feature extractor.
3️⃣ **Fine-tuning** – selectively unfreeze layers and adjust weights.
4️⃣ **Utilities** – dataset adaptation, metrics, evaluation.
5️⃣ **Experiments** – compare feature extraction vs fine-tuning performance.

---

## 🧱 Folder Structure

```
08_Transfer_Learning/
├── feature_extraction.py       # Using pre-trained models as fixed feature extractors
├── fine_tuning.py              # Unfreezing layers and fine-tuning pre-trained models
├── pretrained_models.py        # Downloading & loading popular pre-trained networks
├── tl_utils.py                 # Helper functions for dataset adaptation & evaluation
└── README.md
```

### ✅ Folder Philosophy

* **Research-oriented**: Implements common TL pipelines from literature.
* **Interview-ready**: Can explain differences between feature extraction & fine-tuning.
* **Production-ready**: Code integrates directly with PyTorch or TensorFlow pipelines.

---

## 📄 File Descriptions

### 🔹 `pretrained_models.py`

* Load popular pre-trained networks: ResNet, VGG, EfficientNet, MobileNet.
* Options for **ImageNet** weights or custom pre-trained weights.
* Useful for both **classification** and **feature extraction**.

**Goal:** 
    Understand how to access and adapt pre-trained models for new tasks.
    Load and manage popular pre-trained models such as ResNet, VGG, EfficientNet, and MobileNet.

**Tips:**

* Ability to change the number of classes for a new task.
* All models are loaded from torchvision.
* For interviews: Explain the difference between architectures and the placement of the last layer.
---

### 🔹 `feature_extraction.py`

* Freeze all layers of a pre-trained model.
* Replace the final layer to match the new task's output classes.
* Train **only the new head**, keeping the feature representations fixed.

**Goal:** 
    Learn how to leverage learned features without retraining the entire network.
    Using the pre-trained model as a Feature Extractor, without changing its weights.

**Tips:** 
* Only the last layer (classifier) ​​is trained for the new task.
* Fast training, less data required.
* Can be used directly in PyTorch DataLoader and Training Loop.

---

### 🔹 `fine_tuning.py`

* Gradually unfreeze layers for training.
* Apply lower learning rates to pre-trained layers.
* Strategies:

  * Unfreeze last block only
  * Progressive unfreezing
  * Layer-wise learning rates

**Goal:** 
    Improve task-specific performance by adapting pre-trained features.
    Unfreeze and retrain some model layers for task-specific adaptation.

**Tips:**

* The last few layers can be unfrozen for fine-tuning.
* A lower learning rate is recommended for pre-trained layers.
* Interview: Explain why unfreeze only the last layers and not the entire network.
---

### 🔹 `tl_utils.py`

* Dataset adaptation utilities (resize, normalization, augmentation).
* Evaluation metrics (accuracy, F1-score, confusion matrix).
* Logging and checkpointing pre-trained model experiments.

**Goal:** 
    Simplify experimentation and ensure reproducibility.
    Auxiliary functions to facilitate the use of Transfer Learning, evaluation, and data management.

**Tips:**

* Makes it easy to evaluate performance and draw confusion matrices.
* Logging and reproducibility for TL experiments.

---

## 🧠 Key Concepts / Notes

| Concept                | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| Feature Extraction     | Freeze pre-trained layers; train only new classifier head         |
| Fine-Tuning            | Unfreeze some layers; adjust weights for task-specific learning   |
| Pre-trained Weights    | Use weights trained on large datasets (ImageNet, COCO, etc.)      |
| Domain Adaptation      | Apply pre-trained models to datasets with different distributions |
| Learning Rate Strategy | Use smaller LR for pre-trained layers, higher LR for new layers   |

**Interview Tips:**

* Explain the difference between **feature extraction** and **fine-tuning**.
* Discuss why **learning rates** for pre-trained layers are smaller.
* Be ready to justify **when TL is beneficial** vs training from scratch.

---

## 🎯 Goals After This Folder

* Confidently use pre-trained models for new tasks.
* Know how to implement both feature extraction and fine-tuning.
* Understand **trade-offs**: training speed vs accuracy.
* Apply TL in **real-world projects** with small datasets.
* Answer TL-related interview questions clearly and concisely.
