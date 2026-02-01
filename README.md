# CV & Deep Learning Library (From Scratch to Practice)

A personal, modular, and well-documented archive for learning and implementing
**Computer Vision and Deep Learning concepts** from scratch using **PyTorch**.

This repository is designed to:
- Deeply understand core algorithms (not just use them)
- Build reusable and clean implementations
- Document learning in a structured, resume-ready format
- Serve as a long-term personal CV/DL reference library

---

## 🎯 Project Goals

- Implement core Deep Learning and Computer Vision building blocks **from scratch**
- Compare custom implementations with **official PyTorch modules**
- Maintain clean, modular, and reusable code
- Document **theory, intuition, and implementation details** for each concept
- Create a strong technical portfolio suitable for:
  - Industry roles
  - Research positions
  - Graduate / PhD applications

---

## 🧠 Learning Philosophy

> *Understand before optimizing.*

Each concept is approached in three stages:
1. **Theory & intuition** (documented in README files)
2. **From-scratch implementation** (minimal abstraction)
3. **PyTorch implementation** (industry-standard usage)

This ensures both **deep understanding** and **practical usability**.

---

## 📂 Repository Structure

```text
CV-DeepLearning-Library/
│
├── 00_Core_Building_Blocks/
│   ├── convolution/
│   ├── normalization/
│   ├── loss_functions/
│   └── optimizers/
│
├── 01_Data_Preprocessing/
│   ├── normalization/
│   ├── batch_normalization/
│   └── data_augmentation/
│
├── 02_Weight_Initialization/
│
├── 03_Regularization/
│
├── 04_Optimizers_and_LR_Schedulers/
│
├── 05_Hyperparameter_Tuning/
│
├── 06_Activations/
│
├── 07_Architectural_Blocks/
│
├── 08_Transfer_Learning/
│
├── 09_Dimension_Analysis/
│
├── 10_Experiments_and_Visualizations/
│
└── README.md


Each folder represents a logical learning stage, progressing from fundamental
operations to higher-level architectural and optimization strategies.

📌 Folder Design Principles

Inside each concept folder, you will typically find:

concept_name/
│
├── README.md              # Theory, intuition, history, and explanations
├── *_from_scratch.py      # Manual implementation
├── *_pytorch.py           # PyTorch equivalent
├── test_*.py              # Simple tests / usage examples
└── __init__.py            # For reusability and imports


This structure allows:

Easy reuse in other projects

Clean imports

Clear separation between learning and production code

🧪 Topics Covered
Core Operations

Convolution (Conv1D / Conv2D / Conv3D)

Pooling layers

Fully Connected layers

Data Processing

Normalization

Batch Normalization

Data Augmentation techniques:

Crop, Resize, Flip, Rotate

Color Jitter

Translation, Shearing

Lens Distortion

Creative augmentations

Training Essentials

Loss functions

Optimizers (SGD, Adam, RMSProp)

Learning Rate Schedulers

Weight Initialization (Xavier, Kaiming)

Regularization (Dropout, Label Smoothing)

Model Design

Activation functions

Feature map visualization

Dimension analysis through layers

Transfer learning and fine-tuning strategies

🔁 Reusability

All modules are designed to be importable:

from cvdl.convolution.conv2d import Conv2DFromScratch


This allows seamless integration into:

New experiments

Larger projects

Research prototypes

🚀 Long-Term Vision

This repository is intended to evolve into:

A personal Deep Learning framework

A teaching and reference resource

A strong technical portfolio showcasing real understanding

📚 References & Inspiration

Deep Learning – Ian Goodfellow

CS231n – Stanford

PyTorch Documentation

Original research papers (LeNet, AlexNet, ResNet, etc.)

## 🧑‍💻 Author

**Amin**  
Computer Engineer | Frontend & Deep Learning Enthusiast  
Focused on Computer Vision, Optimization, and Deep Learning Systems  

🔗 **LinkedIn:**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aminmilantash)

💻 **GitHub Resume / Portfolio:**  
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white)](https://github.com/MAmin-Milantash)
