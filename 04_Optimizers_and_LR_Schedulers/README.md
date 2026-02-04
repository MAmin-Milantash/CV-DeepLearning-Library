## 🎯 What is an Optimizer?

An **Optimizer** defines **how model parameters (weights & biases) are updated** during training.

At each training step:

weights = weights - learning_rate × gradient


But **how exactly gradients are used** → that’s the optimizer’s job.

---

## 🧠 Why Optimizers Matter (Intuition)

Two models with:

* same architecture
* same data
* same initialization

can behave **completely differently**
→ just because they use **different optimizers**.

Optimizers control:

* speed of convergence
* stability of training
* ability to escape bad local minima
* sensitivity to noisy gradients

---

## 🧭 Why This Folder Comes After Regularization?

Correct learning order:

1. Data is prepared
2. Weights are initialized
3. Overfitting is controlled (Regularization)
4. **Now we decide how learning actually happens** ← this folder
5. Learning rate scheduling refines the process

---

## 🧱 Folder Structure

04_Optimizers_and_LR_Schedulers/
├── optimizers_from_scratch.py
├── optimizers_torch.py
├── lr_schedulers_from_scratch.py
├── lr_schedulers_torch.py
└── README.md


---

## 🪜 Learning Order Inside This Folder

1️⃣ Basic Gradient Descent  
2️⃣ Momentum-based Optimizers  
3️⃣ Adaptive Optimizers (AdaGrad, RMSProp, Adam)  
4️⃣ Learning Rate Schedulers  

---

## 1️⃣ Gradient Descent Family (Core Idea)

### 🔹 Vanilla Gradient Descent (SGD)

Update rule:

w = w - lr * grad


Problems:

* slow convergence
* sensitive to learning rate
* oscillations in narrow valleys

> Every advanced optimizer is built on top of SGD.

---

## 2️⃣ Momentum-Based Optimizers

### 🔹 Momentum

Idea: Remember previous gradients to move more smoothly.

Update:

v = βv + grad
w = w - lr * v


Benefits:

* faster convergence
* reduced oscillation
* smoother trajectory

### 🔹 Nesterov Accelerated Gradient (NAG)

Looks ahead before computing gradient.  
Benefit: better correction, more stable convergence.

---

## 3️⃣ Adaptive Learning Rate Optimizers

These optimizers **adapt learning rate per parameter**.

### 🔹 AdaGrad

* Accumulates squared gradients
* Rare features get larger updates
* Learning rate decays too fast ❌

### 🔹 RMSProp

* Fixes AdaGrad’s decay issue
* Uses exponential moving average
* Very popular for RNNs

### 🔹 Adam (Most Used)

Combines Momentum (1st moment) & RMSProp (2nd moment)  
Tracks mean & variance of gradients  

Why popular:

* fast convergence
* robust defaults
* works well in most cases

---

## 4️⃣ Learning Rate Schedulers

A single learning rate is rarely optimal.

We want:

* large LR → fast early learning
* small LR → fine convergence later

### 🔹 Common Schedulers

| Scheduler         | Idea                        |
| ----------------- | --------------------------- |
| Step Decay        | Drop LR every N epochs      |
| Exponential Decay | Smooth decay                |
| Cosine Annealing  | Periodic smooth decay       |
| Reduce on Plateau | Reduce when val loss stalls |
| Warm-up           | Start small, then increase  |

---

## 📄 File Responsibilities

### 🔹 optimizers_from_scratch.py

* Manual implementation of:
  * SGD
  * Momentum
  * RMSProp
  * Adam  
* Goal: deep mathematical understanding

### 🔹 optimizers_torch.py

* PyTorch implementations:
  * `torch.optim.SGD`
  * `Adam`
  * `RMSprop`  
* Goal: real-world usage

### 🔹 lr_schedulers_from_scratch.py

* Manual implementation:
  * Step decay
  * Exponential decay
  * Cosine schedule  
* Understand learning dynamics

### 🔹 lr_schedulers_torch.py

* PyTorch schedulers:
  * `StepLR`
  * `ExponentialLR`
  * `ReduceLROnPlateau`
  * `CosineAnnealingLR`

---

## 🎯 Key Takeaways

After finishing this folder, you will know:

* why SGD alone is rarely enough
* how momentum accelerates learning
* why Adam works so well
* when adaptive optimizers fail
* how LR scheduling improves convergence
* how to choose optimizer + scheduler in practice

---

## 🔥 Interview-Level Insights

* Why Adam sometimes generalizes worse than SGD?  
* Why warm-up is critical for large models?  
* When should learning rate be reduced?  
* Is Adam always the best choice?