# 📁 09_Dimension_Analysis

## 🧭 Purpose

This folder focuses on **analyzing the dimensionality of data** throughout a neural network pipeline. Understanding shapes, dimensions, and latent representations is critical for:

* Debugging mismatched inputs in layers
* Visualizing feature map transformations
* Understanding bottlenecks and expansion in networks
* Designing multi-branch and complex architectures
* Exploring latent spaces via Autoencoders, PCA, and t-SNE

It’s **especially useful for CNNs, ResNets, Inception modules, attention-based networks, and generative models**.

---

## 🧭 Why Dimension Analysis Matters

* Ensures that **layer inputs/outputs match** expected shapes
* Detects **broadcasting issues, flattening mistakes, or reshaping errors**
* Facilitates **latent space analysis and feature visualization**
* Critical for **model debugging and architectural design**

---

## 🪜 Recommended Learning Order

1️⃣ Input data shape exploration and summary  
2️⃣ Feature map dimension tracing through Dense/Conv/Residual blocks  
3️⃣ Flattening, reshaping, and concatenation handling  
4️⃣ Multi-branch and skip connection dimension verification  
5️⃣ Latent space analysis with Autoencoders (undercomplete, overcomplete, denoising, contractive, sparse, variational)  
6️⃣ Dimensionality reduction & visualization with PCA & t-SNE  
7️⃣ Utility functions for automated shape logging and visualization  

---

## 🧱 Folder Structure

```

09_Dimension_Analysis/
├── input_analysis.py              # Explore raw input shapes, channels, batches
├── conv_feature_maps.py           # Trace Conv layers and output dimensions
├── residual_dimensions.py         # Check skip connections & addition layers
├── inception_dimensions.py        # Verify multi-branch concatenation outputs
├── attention_dims.py              # Track shapes in self-attention modules
├── flatten_and_reshape.py         # Utilities for flattening and reshaping tensors
├── pca_analysis.py                # Principal Component Analysis for latent/feature space
├── tsne_analysis.py               # t-SNE for high-dimensional feature visualization
├── autoencoder_analysis.py        # Undercomplete & Overcomplete Autoencoders
├── autoencoder_denoising.py       # Denoising Autoencoder (DAE)
├── autoencoder_contractive.py     # Contractive Autoencoder (CAE)
├── autoencoder_sparse.py          # Sparse Autoencoder
├── autoencoder_variational.py     # Variational Autoencoder (VAE)
├── utils.py                       # Helper functions: print shapes, visualize latent space, log errors
└── README.md

```

---

## 📄 File Descriptions

### 🔹 `input_analysis.py`
* Analyze input data shapes: images, sequences, or tabular features.
* Compute batch, channel, height, width (CNN) or features (MLP).
* Detect inconsistencies before feeding into the network.

### 🔹 `conv_feature_maps.py`
* Trace Conv2D/Conv3D layers.
* Compute output shapes given kernel, stride, padding, dilation.
* Visualize feature map dimensions across the network.

### 🔹 `residual_dimensions.py`
* Check skip connections for dimension compatibility.
* Supports addition or concatenation in residual blocks.

### 🔹 `inception_dimensions.py`
* Analyze multi-branch outputs (1x1, 3x3, 5x5 convs)
* Merge branch outputs and verify channel dimensions.

### 🔹 `attention_dims.py`
* Track query, key, value tensor shapes in self-attention layers.
* Verify head splitting, concatenation, and projection dimensions.

### 🔹 `flatten_and_reshape.py`
* Utilities for flattening Conv features to feed Dense layers
* Reshape tensors for concatenation or multi-branch integration

### 🔹 `pca_analysis.py`
* Perform **Principal Component Analysis (PCA)** on features or latent space.
* Reduce dimensionality for visualization.
* Identify major variance directions.

### 🔹 `tsne_analysis.py`
* Perform **t-SNE** on high-dimensional features.
* Visualize clusters and latent manifolds in 2D/3D space.

### 🔹 `autoencoder_analysis.py`
* Implements **undercomplete & overcomplete autoencoders**.
* Analyze reconstruction and latent space dimensions.
* Goal: understand bottlenecks and compression.

### 🔹 `autoencoder_denoising.py`
* Implements **Denoising Autoencoder (DAE)**.
* Input: noisy version of original data, output: clean reconstruction.
* Goal: learn robust features.

### 🔹 `autoencoder_contractive.py`
* Implements **Contractive Autoencoder (CAE)**.
* Adds Jacobian-based regularization to latent representation.
* Goal: encourage robustness to small input perturbations.

### 🔹 `autoencoder_sparse.py`
* Implements **Sparse Autoencoder**.
* Regularizes latent activations to be sparse.
* Goal: feature selection and interpretability.

### 🔹 `autoencoder_variational.py`
* Implements **Variational Autoencoder (VAE)**.
* Latent space is probabilistic (mean & variance vectors).
* Goal: generative modeling and smooth latent interpolation.

### 🔹 `utils.py`
* Print layer shapes dynamically.
* Visualize latent spaces & reconstructions.
* Log errors and latent statistics.
* Support reproducible experiments.

---

## 🧠 Key Concepts / Notes

| Topic                 | Purpose / Benefit                                  |
| --------------------- | -------------------------------------------------- |
| Input Analysis        | Detect incompatible shapes early                   |
| Conv Feature Maps     | Track spatial and channel dimensions               |
| Residual Connections  | Ensure addition/skip works correctly               |
| Multi-branch Networks | Verify concatenation, channel aggregation          |
| Attention Mechanisms  | Track query/key/value shapes and head projections  |
| Flatten / Reshape     | Smooth transition between Conv and Dense layers    |
| PCA / t-SNE           | Dimensionality reduction & visualization           |
| Autoencoders          | Latent space analysis, robust feature learning     |
| Denoising             | Noise removal, robust representations              |
| Contractive           | Smooth latent manifolds                             |
| Sparse                | Feature selection, interpretability               |
| Variational           | Generative modeling, probabilistic latent space   |

---

## 🔹 Autoencoder Comparison

| Autoencoder Type       | Latent Dim | Regularization / Constraint                 | Strengths                                               | Weaknesses                                              | Typical Use Cases                                      |
| --------------------- | ---------- | ------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------- |
| Undercomplete          | < Input    | Bottleneck constraint                        | Simple, effective compression                            | May lose fine details                                   | Basic compression, dimensionality reduction          |
| Overcomplete           | ≥ Input    | None (latent > input)                        | Captures more info, flexible latent                    | Risk of learning identity mapping                        | Feature extraction, pretraining                       |
| Denoising (DAE)        | < Input    | Bottleneck + input corruption               | Robust to noise, learns meaningful representations     | Requires noise modeling                                 | Image denoising, robust embeddings                   |
| Contractive (CAE)      | < Input    | Jacobian penalty on latent                   | Smooth latent space, resistant to small perturbations  | More complex, extra computation                         | Robust representation, feature invariance            |
| Sparse                 | < Input    | L1 penalty on latent activations             | Encourages sparse, interpretable features              | Hyperparameter tuning needed                             | Feature selection, interpretability                 |
| Variational (VAE)      | Probabilistic | KL divergence regularization                 | Generative modeling, smooth latent manifold           | Complex training, stochastic output                      | Generative models, latent space interpolation       |

**Insights:**

* Bottleneck-based Autoencoders (undercomplete, denoising, contractive, sparse) focus on robust, compressed representation.  
* Overcomplete Autoencoders may memorize input but useful with sparsity or denoising constraints.  
* Variational Autoencoders provide **probabilistic latent spaces**, ideal for generative tasks.  
* Choice depends on **application goal**: compression, robustness, feature extraction, or generative modeling.  

---

## 🎯 Goals After This Folder

* Understand **how each layer changes tensor shapes**.
* Detect and prevent **dimensionality errors**.
* Analyze **latent space and feature embeddings**.
* Build **robust, compressed, and generative representations**.
* Compare **strengths and weaknesses of all major Autoencoder types**.
* Prepare for **real-world networks, debugging, and research experiments**.
```
