📁 06_Activation_Functions
🧭 Why This Folder?

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without them, the network is just a linear combination of inputs and cannot approximate complex functions.

After the previous folders (Data Preprocessing → Weight Initialization → Regularization → Optimizers → Hyperparameter Tuning), it is now essential to understand how activations affect forward and backward propagation, gradient flow, and convergence.

🧱 Folder Structure
06_Activation_Functions/
├── sigmoid_from_scratch.py
├── tanh_from_scratch.py
├── relu_from_scratch.py
├── leaky_relu_from_scratch.py
├── activations_torch.py
├── activations_utils.py
└── README.md

🪜 Learning Order Inside This Folder

1️⃣ Sigmoid → classic, simple, shows saturation problem
2️⃣ Tanh → zero-centered, reduces bias
3️⃣ ReLU → fast convergence, standard in modern architectures
4️⃣ Leaky ReLU → solves dying ReLU problem
5️⃣ Torch built-in activations → practical implementation
6️⃣ Utilities → visualization and gradient check

📄 File Descriptions
🔹 sigmoid_from_scratch.py

Purpose:
Manual implementation of the Sigmoid activation function, including forward and backward passes.


**Sigmoid Activation Function**

Forward:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Derivative (backward):

$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

Output range: (0,1)

Saturates for large |x| → vanishing gradient problem

Mostly used in binary classification outputs

🔹 tanh_from_scratch.py

Purpose:
Manual implementation of the Tanh activation function, including forward and backward passes.


**Tanh Activation Function**

Forward:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

Derivative (backward):

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)
$$

Output range: (-1,1), zero-centered

Saturates for large |x| → vanishing gradient possible

Often better than Sigmoid in hidden layers

🔹 relu_from_scratch.py

Purpose:
Manual implementation of ReLU: 

**ReLU Activation Function**

Forward:

$$
\text{ReLU}(x) = \max(0, x)
$$

Derivative (backward):

$$
\frac{d}{dx}\text{ReLU}(x) =
\begin{cases} 
1 & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$


Forward + backward

Fast computation, sparse activations

Popular in modern deep learning

Demonstrates dying ReLU problem (neurons stuck at 0)

🔹 leaky_relu_from_scratch.py

Purpose:
Manual implementation of Leaky ReLU: 


**Leaky ReLU Activation Function**

Forward:

$$
\text{LeakyReLU}(x) =
\begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \le 0
\end{cases}
$$

Derivative (backward):

$$
\frac{d}{dx}\text{LeakyReLU}(x) =
\begin{cases} 
1 & \text{if } x > 0 \\
\alpha & \text{if } x \le 0
\end{cases}
$$

keeps gradient alive for negative inputs

Solves dying ReLU problem

Backward pass implemented manually

🔹 activations_torch.py

Purpose:
PyTorch built-in implementations for practical use.

Includes:

nn.Sigmoid

nn.Tanh

nn.ReLU

nn.LeakyReLU

nn.Softmax

Key Concepts:

Forward only

Ready for plugging into real models

Shows difference between scratch vs built-in

🔹 activations_utils.py

Purpose:
Utility functions for plotting and comparing activations and their gradients.

Examples:

plot_activation_curve(func) → shows output vs input

plot_gradient(func) → shows derivative magnitudes

Compare Sigmoid, Tanh, ReLU, Leaky ReLU side-by-side

Helps visualize vanishing/exploding gradient issues

🎯 Key Takeaways

Activation functions introduce non-linearity

Choice of activation affects gradient flow and convergence speed

ReLU is standard in hidden layers; Sigmoid/Tanh mostly for outputs

Leaky ReLU fixes dying neuron issues

Visualizing activations and gradients is crucial for debugging and interviews