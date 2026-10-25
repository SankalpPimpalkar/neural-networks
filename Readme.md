# Neural Networks & Autograd Engine from Scratch

A lightweight, scalar-valued automatic differentiation (autograd) engine and deep learning framework built from first principles in Python, featuring computational graph visualization, modular neural network layers, parameter tracking, and gradient-based optimization.

Inspired by Andrej Karpathy's `micrograd`, this project implements the mathematical foundations of deep learning: dynamic computational graphs, exact reverse-mode automatic differentiation via the chain rule with topological sorting, modular neural network components (`Neuron`, `Layer`, `MLP`), parameter tracking, loss minimization, and validation against PyTorch.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
  - [1. Scalar Autodiff Engine (`value.py`)](#1-scalar-autodiff-engine-valuepy)
  - [2. Neural Network Architecture & Parameter Tracking (`neural-network.py`)](#2-neural-network-architecture--parameter-tracking-neural-networkpy)
  - [3. Function Decomposition & Gradient Accuracy (`micrograd.py`)](#3-function-decomposition--gradient-accuracy-microgradpy)
  - [4. PyTorch Validation & Benchmarking (`torch-micrograd.py`)](#4-pytorch-validation--benchmarking-torch-microgradpy)
  - [5. Computation Graph Visualization (`dd.py`)](#5-computation-graph-visualization-ddpy)
  - [6. Vectorized Feedforward Baseline (`nn.py`)](#6-vectorized-feedforward-baseline-nnpy)
  - [7. Finite Difference Verification (`functions.py`)](#7-finite-difference-verification-functionspy)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
  - [Scalar Arithmetic & Autodiff](#scalar-arithmetic--autodiff)
  - [Training an MLP with Parameter Tracking](#training-an-mlp-with-parameter-tracking)
  - [Visualizing Computation & Loss Graphs](#visualizing-computation--loss-graphs)
  - [Validating Against PyTorch](#validating-against-pytorch)
- [How Backpropagation Works](#how-backpropagation-works)
- [Project Roadmap & Commit History](#project-roadmap--commit-history)

---

## Project Overview

Modern deep learning frameworks (PyTorch, JAX, TensorFlow) abstract away graph construction and reverse-mode automatic differentiation. 

This repository exposes the underlying mechanics from scratch:
1. **Dynamic Scalar Graphing**: Mathematical expressions dynamically generate a Directed Acyclic Graph (DAG) tracking scalar values, operation history, and parent dependencies.
2. **First-Principles Calculus**: Each operator defines its exact analytical derivative closure ($d(\text{out})/d(\text{in})$) adhering to the chain rule.
3. **Atomic Decomposition**: Complex activations like $\tanh(x)$ can be decomposed into elementary operations ($\exp$, division via negative power, subtraction, addition) while maintaining exact gradient precision.
4. **PyTorch-style Neural Networks**: Object-oriented `Neuron`, `Layer`, and `MLP` modules built entirely from scalar `Value` nodes.
5. **Parameter Tracking & Training**: Recursive `.parameters()` collection across all layers, computing loss, backpropagation, and updating weights using gradient descent.
6. **Visual DAG Inspection**: Interactive SVG rendering showing values, accumulated gradients, and operation nodes at every intermediate step.

---

## Key Features

- **Full Algebraic Operator Overloading**: Supports `+`, `-`, `*`, `/`, `**` (powers), unary negation (`-`), right-side operations (`__radd__`, `__rmul__`), and seamless interoperability with Python native numbers (`int`, `float`).
- **Activation Functions**: Native `tanh()` and exponential `exp()` with exact local derivatives.
- **Topological Sorting**: Uses depth-first search (`build_topo`) post-order traversal to sequence graph evaluation before backpropagation.
- **Gradient Accumulation**: Safe multivariate chain rule gradient accumulation (`+=`) ensuring nodes with multiple output branches compute correct total derivatives.
- **Modular Neural Network Components**:
  - `Neuron`: Linear combination $w^T x + b$ followed by $\tanh$ activation.
  - `Layer`: Collection of parallel neurons mapping an input vector to layer outputs.
  - `MLP`: Multi-Layer Perceptron architecture with configurable input and layer sizes (e.g. `MLP(3, [4, 4, 1])`).
- **Parameter Management**: `parameters()` method on `Neuron`, `Layer`, and `MLP` to retrieve all trainable weights and biases.
- **Loss Computation & Training**: Squared error loss tracking, gradient descent parameter updates (`p.data += -step_sz * p.grad`), and gradient zeroing (`p.grad = 0.0`).
- **PyTorch Parity**: Verified against PyTorch's native `torch.Tensor` autograd engine with double-precision accuracy.
- **Graphviz Visualization**: SVG export functions (`dd` and `draw`) to inspect the computation graph and loss expressions.

---

## Core Components

### 1. Scalar Autodiff Engine (`value.py`)

The `Value` class is the central scalar node in the computational graph:
- **`data`**: Float holding the node's scalar value.
- **`grad`**: Accumulated derivative $\frac{\partial L}{\partial v}$, initialized to `0.0`.
- **`_prev`**: Set of parent `Value` nodes.
- **`_op`**: String representation of the operation that produced this node (`'+'`, `'*'`, `'**'`, `'exp'`, `'tanh'`).
- **`_backward`**: Closure computing local gradients and propagating them to parents via the chain rule.

#### Supported Operations & Local Derivatives

| Operation | Implementation | Mathematical Gradient ($\frac{\partial L}{\partial x}$) |
|---|---|---|
| **Addition** | `a + b`, `b + a` (`__add__`, `__radd__`) | $\frac{\partial (a+b)}{\partial a} = 1.0 \cdot \text{out.grad}$ |
| **Negation** | `-a` (`__neg__`) | $\frac{\partial (-a)}{\partial a} = -1.0 \cdot \text{out.grad}$ |
| **Subtraction** | `a - b` (`__sub__`) | Implemented as `a + (-b)` |
| **Multiplication** | `a * b`, `b * a` (`__mul__`, `__rmul__`) | $\frac{\partial (a \cdot b)}{\partial a} = b \cdot \text{out.grad}$ |
| **Power** | `a ** k` (`__pow__`) | $\frac{\partial (a^k)}{\partial a} = k \cdot a^{k-1} \cdot \text{out.grad}$ |
| **Division** | `a / b` (`__truediv__`) | Implemented as `a * (b ** -1)` |
| **Exponential** | `a.exp()` | $\frac{\partial (e^a)}{\partial a} = e^a \cdot \text{out.grad}$ |
| **Hyperbolic Tangent** | `a.tanh()` | $\frac{\partial \tanh(a)}{\partial a} = (1 - \tanh^2(a)) \cdot \text{out.grad}$ |

All binary methods automatically convert raw Python numbers (`int`, `float`) into `Value` instances.

---

### 2. Neural Network Architecture & Parameter Tracking (`neural-network.py`)

Implements an object-oriented deep learning API powered by scalar `Value` objects:

#### `Neuron(nin)`
- Initializes $n_{\text{in}}$ weights and a single bias randomly from the uniform distribution $\mathcal{U}(-1, 1)$.
- Forward call `__call__(x)` computes:
  $$\text{out} = \tanh\left(\sum_{i=1}^{n_{\text{in}}} w_i x_i + b\right)$$
  using Python's `sum(..., self.b)`.
- `parameters()`: Returns the list of trainable weights and bias: `self.w + [self.b]`.

#### `Layer(nin, nout)`
- Contains `nout` independent `Neuron` instances, each taking `nin` inputs.
- Forward call `__call__(x)` returns a single `Value` if `nout == 1`, or a list of `Value` objects if `nout > 1`.
- `parameters()`: Collects and returns parameters from all neurons in the layer.

#### `MLP(nin, nouts)`
- Multi-Layer Perceptron taking input dimension `nin` and a list of output dimensions `nouts`.
- Example: `MLP(3, [4, 4, 1])` defines 3 inputs, two hidden layers of 4 neurons each, and 1 output neuron (total: 41 parameters).
- `parameters()`: Recursively aggregates parameters across all layers:
  ```python
  def parameters(self):
      params = []
      for l in self.layers:
          params.extend(l.parameters())
      return params
  ```

#### Loss Tracking & Training Loop
`neural-network.py` trains the network on a toy 4-sample dataset:
```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
n = MLP(3, [4, 4, 1])

# Initial predictions and loss
ypred = [n(x) for x in xs]
loss = sum([(yc - yd)**2 for yc, yd in zip(ypred, ys)])
print("Initial Loss: ", loss)

# Optimization loop: update parameters and zero gradients
step_sz = 0.01
for i in range(100):
    for p in n.parameters():
        p.data += -step_sz * p.grad

    for p in n.parameters():
        p.grad = 0.0

    ypred = [n(x) for x in xs]
    loss = sum([(yc - yd)**2 for yc, yd in zip(ypred, ys)])

print("Final Loss: ", loss)
draw(loss)
```

---

### 3. Function Decomposition & Gradient Accuracy (`micrograd.py`)

Demonstrates that composite activation functions can be broken down into elementary operations without losing gradient precision.

Instead of treating $\tanh$ as an opaque black box, it is decomposed into atomic exponential, power, and arithmetic primitives:
```python
e = (2 * n).exp()
out = (e - 1) / (e + 1)
out.label = "O"

out.backward()
```

The autograd engine traverses each step:
1. Division via power of $-1$: `(e + 1) ** -1`
2. Subtraction: `e - 1`
3. Exponentiation: `(2 * n).exp()`
4. Scalar product: `2 * n`

All weights and biases receive gradients matching the atomic `tanh()` output and PyTorch down to floating-point precision limits.

---

### 4. PyTorch Validation & Benchmarking (`torch-micrograd.py`)

A direct benchmark verifying gradient parity against PyTorch's native autograd engine (`torch.Tensor` with `requires_grad=True`):

```python
# Equivalent PyTorch graph:
n = x1*w1 + x2*w2 + b
output = torch.tanh(n)
output.backward()
```

#### Gradient Parity Comparison:
| Variable | PyTorch Gradient | Custom `Value` Gradient | Absolute Difference |
|---|---|---|---|
| **$x_1$** | `-1.50079251` | `-1.50079260` | $< 10^{-7}$ |
| **$w_1$** | `1.00052834` | `1.00052840` | $< 10^{-7}$ |
| **$x_2$** | `0.50026417` | `0.50026420` | $< 10^{-7}$ |
| **$w_2$** | `0.00000000` | `0.00000000` | `0.0` |
| **$b$**   | `0.50026417` | `0.50026420` | $< 10^{-7}$ |

---

### 5. Computation Graph Visualization (`dd.py`)

Visualizes computational DAGs using [Graphviz](https://graphviz.org/):
- **`trace(root)`**: Traverses backward from the root to collect all reachable nodes and directed edges.
- **`dd(root)`**: Builds a Graphviz `Digraph` with nodes rendered with record shapes `{ label | data %.4f | grad %.4f }` and distinct operation nodes (`+`, `*`, `**-1`, `exp`, etc.).
- **`draw(root, path="draw")`**: Helper function that directly renders the computation graph to an SVG file (`draw.svg`).

#### Computation Graph Preview (`draw.svg`)

![Computation Graph Preview](./draw.svg)

---

### 6. Vectorized Feedforward Baseline (`nn.py`)

A pure NumPy implementation demonstrating feedforward computation with matrix operations:
- Layer architecture: `[2, 3, 3, 1]`.
- Implements $\text{output} = \sigma(X W + b)$ across layers using vectorized matrix products (`@`).

---

### 7. Finite Difference Verification (`functions.py`)

Implements numerical gradient checking using small finite-difference perturbations ($h = 0.001$):

$$\frac{df(x)}{dx} \approx \frac{f(x + h) - f(x)}{h}$$

Used to sanity-check analytical gradients produced by `Value.backward()`.

---

## Project Structure

```text
.
├── value.py             # Core autograd engine (Value class, operators, topological DFS)
├── neural-network.py    # Modular deep learning framework: Neuron, Layer, MLP, training loop
├── micrograd.py         # Neuron forward/backward pass with decomposed tanh & SVG rendering
├── torch-micrograd.py   # Benchmark script comparing custom autograd directly against PyTorch
├── dd.py                # Graphviz computation graph renderer (trace, dd, and draw)
├── nn.py                # Vectorized feedforward MLP baseline using NumPy
├── functions.py         # Finite difference numerical gradient sanity checks
├── equations.py         # Sample polynomial test equations
├── draw.svg             # Rendered computational graph output
├── requirements.txt     # Python package dependencies
└── Readme.md            # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9+
- [Graphviz](https://graphviz.org/download/) (for graph visualization)
  - **macOS**: `brew install graphviz`
  - **Ubuntu / Debian**: `sudo apt-get install graphviz`
  - **Windows**: `winget install Graphviz` or `choco install graphviz`

### 2. Environment Setup

```bash
# Clone repository
git clone https://github.com/SankalpPimpalkar/neural-networks.git
cd neural-networks

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install graphviz torch
```

---

## Usage Examples

### Scalar Arithmetic & Autodiff

```python
from value import Value

a = Value(2.0, label='a')
b = Value(-3.0, label='b')

# Rich operators and scalar mixing
c = a + 10.0         # __add__ with float
d = 2.0 * b          # __rmul__ with float
e = (c / d) ** 2     # Division and exponentiation
L = e.exp()          # Exponential function
L.label = 'L'

L.backward()

print(f"L data: {L.data:.4f}")
print(f"dL/da:  {a.grad:.4f}")
print(f"dL/db:  {b.grad:.4f}")
```

### Training an MLP with Parameter Tracking

```python
from value import Value
from dd import draw
import random

# Import classes from neural-network.py (Neuron, Layer, MLP)
from neural_network import MLP  # or directly execute neural-network.py

model = MLP(3, [4, 4, 1])
print("Total Parameters:", len(model.parameters()))

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# Training loop
step_sz = 0.05
for epoch in range(100):
    # 1. Forward pass
    ypred = [model(x) for x in xs]
    loss = sum([(yc - yd)**2 for yc, yd in zip(ypred, ys)])

    # 2. Backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # 3. Parameter update (gradient descent)
    for p in model.parameters():
        p.data += -step_sz * p.grad

print(f"Final Loss: {loss.data:.4f}")
```

### Visualizing Computation & Loss Graphs

```python
from value import Value
from dd import draw

x1 = Value(2.0, label='x1')
w1 = Value(-3.0, label='w1')
b = Value(6.881, label='b')

n = x1 * w1 + b
n.label = 'n'

# Decomposed hyperbolic tangent
e = (2 * n).exp()
out = (e - 1) / (e + 1)
out.label = 'out'

out.backward()

# Render graph to SVG (saves as 'draw.svg')
draw(out, path="draw")
```

### Validating Against PyTorch

Run the benchmark script to compare gradients side-by-side:

```bash
python torch-micrograd.py
```

---

## How Backpropagation Works

1. **Local Gradient Closures**:
   Every operation creates an output node and binds a `_backward()` closure that knows how to calculate local derivatives and propagate `out.grad` backward to its inputs.
   ```python
   def __mul__(self, other):
       other = other if isinstance(other, Value) else Value(other)
       out = Value(self.data * other.data, (self, other), '*')
       def _backward():
           self.grad += other.data * out.grad
           other.grad += self.data * out.grad
       out._backward = _backward
       return out
   ```
2. **Topological Order (DFS)**:
   Graphs can branch, recombine, and share variables. The `backward()` method computes a topological sort via Depth-First Search (`build_topo`) ensuring no node computes its gradients until all child nodes depending on it have processed theirs:
   ```python
   topo = []
   visited = set()
   def build_topo(v):
       if v not in visited:
           visited.add(v)
           for child in v._prev:
               build_topo(child)
           topo.append(v)
   ```
3. **Multivariate Chain Rule (Accumulation)**:
   Gradients are accumulated using `+=` rather than assigned (`=`). If variable $x$ flows into multiple operations ($y = f(x)$ and $z = g(x)$), the total derivative sums the contributions:
   $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial x} + \frac{\partial L}{\partial z}\frac{\partial z}{\partial x}$$

---

## Project Roadmap & Commit History

This project was built iteratively from scratch. Below is the complete development roadmap matching the project's commit history:

- **1. Initial Setup & Foundation** (`83e84d9`)
  - Project initialized with base workspace structure and environment setup.

- **2. Core Value Representation & Graph Visualization** (`27aed56`)
  - Implemented the foundational `Value` class for scalar data representation.
  - Added test mathematical equations (`parabola` in `equations.py`).
  - Introduced graph tracing (`trace`) and DAG visualization (`dd`) using Graphviz to render computation graphs to SVG.

- **3. Automatic Differentiation & Topological Sort** (`6b1ed23`)
  - Implemented reverse-mode automatic differentiation in `Value.backward()`.
  - Added topological sorting via recursive depth-first search (`build_topo`) to sequence gradient evaluation correctly across the DAG.
  - Implemented local backward closures for addition (`+`), multiplication (`*`), and hyperbolic tangent (`tanh`).
  - Added numerical gradient testing via finite difference approximations (`test_grad` in `functions.py`).
  - Implemented the artificial neuron computation graph in `micrograd.py`.

- **4. Project Documentation & Knowledge Base** (`665802b`)
  - Documented the mathematical mechanics, component architecture, and visualization pipeline.

- **5. Extended Arithmetic Operator Overloading** (`c71708e`, `3837b34`)
  - Added support for non-`Value` types (`int`, `float`) across binary arithmetic operations.
  - Implemented power (`**`) with its power rule derivative and division (`/`) via negative powers.
  - Implemented natural exponential (`exp()`) with exponential derivative.
  - Added unary negation (`__neg__`) and subtraction (`__sub__`).
  - Demonstrated composite activation decomposition in `micrograd.py` ($\tanh$ constructed from $\exp$, powers, subtraction, and division).
  - Validated gradient parity against PyTorch in `torch-micrograd.py`.

- **6. Neural Network Architecture, Parameter Tracking & Loss Optimization** (`985b502`)
  - Implemented object-oriented neural network building blocks: `Neuron`, `Layer`, and `MLP` in `neural-network.py`.
  - Added `parameters()` method across `Neuron`, `Layer`, and `MLP` for centralized weight and bias management.
  - Implemented forward pass evaluation and squared error loss calculation on multi-sample training data.
  - Built an iterative training loop updating parameters using gradient descent (`p.data += -step_sz * p.grad`) and gradient zeroing (`p.grad = 0.0`).
  - Added convenient `draw(root, path)` utility in `dd.py` to directly render computation and loss graphs to SVG.
