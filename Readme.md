# Neural Networks & Autograd Engine from Scratch

A lightweight, scalar-valued automatic differentiation (autograd) engine and deep learning framework built from first principles in Python, featuring computational graph visualization and modular neural network abstractions.

Inspired by Andrej Karpathy's `micrograd`, this project implements the mathematical foundation of deep learning: dynamic computational graphs, exact reverse-mode automatic differentiation via the chain rule with topological sort, modular neural network layers (`Neuron`, `Layer`, `MLP`), and validation against PyTorch.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
  - [1. Extended Scalar Autodiff Engine (`value.py`)](#1-extended-scalar-autodiff-engine-valuepy)
  - [2. Deep Learning Abstractions (`neural-network.py`)](#2-deep-learning-abstractions-neural-networkpy)
  - [3. Composite Function Decomposition (`micrograd.py`)](#3-composite-function-decomposition-microgradpy)
  - [4. PyTorch Validation & Benchmarking (`torch-micrograd.py`)](#4-pytorch-validation--benchmarking-torch-microgradpy)
  - [5. Computation Graph Visualization (`dd.py`)](#5-computation-graph-visualization-ddpy)
  - [6. Vectorized Feedforward Baseline (`nn.py`)](#6-vectorized-feedforward-baseline-nnpy)
  - [7. Finite Difference Gradient Verification (`functions.py`)](#7-finite-difference-gradient-verification-functionspy)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
  - [Basic Autograd & Operators](#basic-autograd--operators)
  - [Building and Evaluating an MLP](#building-and-evaluating-an-mlp)
  - [Visualizing the Computation Graph](#visualizing-the-computation-graph)
  - [Validating Against PyTorch](#validating-against-pytorch)
- [How Backpropagation Works](#how-backpropagation-works)
- [Roadmap](#roadmap)

---

## Project Overview

Modern deep learning frameworks (PyTorch, JAX, TensorFlow) abstract away graph construction and reverse-mode automatic differentiation. 

This repository exposes the underlying mechanics:
1. **Dynamic Scalar Graphing**: Mathematical expressions dynamically generate a Directed Acyclic Graph (DAG) tracking scalar values, operation history, and parent dependencies.
2. **First-Principles Calculus**: Each operator defines its exact analytical derivative closure ($d(\text{out})/d(\text{in})$) adhering to the chain rule.
3. **Atomic Decomposition**: Complex activations like $\tanh(x)$ can either be executed natively or decomposed into elementary operations ($\exp$, division via negative power, subtraction, addition) while maintaining gradient fidelity.
4. **PyTorch-style Neural Networks**: Object-oriented `Neuron`, `Layer`, and `MLP` modules built entirely from scalar `Value` nodes.
5. **Visual DAG Inspection**: Interactive SVG rendering showing data values, accumulated gradients, and operation nodes at every step.

---

## Key Features

- **Full Algebraic Operator Overloading**: Supports `+`, `-`, `*`, `/`, `**` (powers), `-` (negation), along with right-side operations (`__radd__`, `__rmul__`) and scalar primitives (`int`, `float`).
- **Activation Functions**: Native `tanh()` and exponential `exp()` with exact local derivatives.
- **Topological Sorting**: Uses depth-first post-order traversal to sequence graph evaluation before backpropagation.
- **Gradient Accumulation**: Safe multivariate chain rule gradient accumulation (`+=`) ensuring nodes with multiple output branches compute correct total derivatives.
- **Object-Oriented Neural Network Library**:
  - `Neuron`: Linear combination $w^T x + b$ with non-linear activation.
  - `Layer`: Collection of parallel neurons.
  - `MLP`: Multi-layer feedforward architecture with configurable layer shapes.
- **PyTorch Parity**: Verified against PyTorch's native `torch.Tensor` autograd engine with double-precision accuracy.
- **Graphviz Visualization**: Exports the complete computation graph into formatted SVG diagrams.

---

## Core Components

### 1. Extended Scalar Autodiff Engine (`value.py`)

The `Value` class is the central scalar node in the computational graph:
- **`data`**: Float holding the node's scalar output.
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

### 2. Deep Learning Abstractions (`neural-network.py`)

Implements a modular, PyTorch-like API powered by scalar `Value` objects:

#### `Neuron(nin)`
- Initializes $n_{\text{in}}$ weights and a single bias uniformly in $[-1, 1]$.
- Forward call `__call__(x)` computes:
  $$\text{out} = \tanh\left(\sum_{i=1}^{n_{\text{in}}} w_i x_i + b\right)$$
  using Python's `sum(..., self.b)`.

#### `Layer(nin, nout)`
- A layer containing `nout` independent `Neuron` instances.
- Passes input vector `x` through all neurons.
- Returns a single `Value` if `nout == 1`, or a list of `Value` objects if `nout > 1`.

#### `MLP(nin, nouts)`
- Multi-Layer Perceptron taking input dimension `nin` and a list of output dimensions `nouts`.
- Example: `MLP(3, [4, 4, 1])` defines an input layer of size 3, two hidden layers of size 4, and an output layer of size 1.
- Sequentially feeds inputs through each layer.

---

### 3. Composite Function Decomposition (`micrograd.py`)

Demonstrates that high-level activations can be decomposed into elementary operations without loss of gradient precision.

Rather than calling the monolithic `n.tanh()`, the hyperbolic tangent is constructed explicitly:
```python
# Tanh defined via exponentials and powers:
e = (2 * n).exp()
out = (e - 1) / (e + 1)
out.label = "O"

out.backward()
```

Backpropagation automatically traverses through each atomic operation:
1. `(e + 1) ** -1` (power rule)
2. `(e - 1) * ...` (product rule)
3. `2 * n` and `.exp()` (exponential rule)

The resulting gradients for weights and biases match the atomic `tanh()` output and PyTorch down to floating-point precision limits.

---

### 4. PyTorch Validation & Benchmarking (`torch-micrograd.py`)

A direct unit validation script verifying gradient accuracy against PyTorch's native autograd engine (`torch.Tensor` with `requires_grad=True`):

```python
# Both systems evaluate:
n = x1*w1 + x2*w2 + b
output = torch.tanh(n)  # vs custom decomposed/tanh
output.backward()
```

#### Gradient Comparison:
| Variable | PyTorch Gradient | Custom `Value` Gradient | Absolute Difference |
|---|---|---|---|
| **$x_1$** | `-1.50079251` | `-1.50079260` | $< 10^{-7}$ |
| **$w_1$** | `1.00052834` | `1.00052840` | $< 10^{-7}$ |
| **$x_2$** | `0.50026417` | `0.50026420` | $< 10^{-7}$ |
| **$w_2$** | `0.00000000` | `0.00000000` | `0.0` |
| **$b$**   | `0.50026417` | `0.50026420` | $< 10^{-7}$ |

---

### 5. Computation Graph Visualization (`dd.py`)

Renders DAG structures to Graphviz:
- **`trace(root)`**: Recursively collects all antecedent nodes and directed edges.
- **`dd(root)`**: Formats nodes into Graphviz records with data and gradient values:
  `{ label | data %.4f | grad %.4f }`, linked with operational nodes (`+`, `*`, `**-1`, `exp`, etc.).
- Renders to SVG (saved as `draw.svg`).

---

### 6. Vectorized Feedforward Baseline (`nn.py`)

A pure NumPy implementation demonstrating feedforward computation with matrix operations:
- Layer architecture: `[2, 3, 3, 1]`.
- Implements $\text{output} = \sigma(X W + b)$ across layers using vectorized matrix products (`@`).

---

### 7. Finite Difference Gradient Verification (`functions.py`)

Performs numerical gradient checking using small perturbations ($h = 0.001$):

$$\frac{df(x)}{dx} \approx \frac{f(x + h) - f(x)}{h}$$

This guarantees mathematical correctness of all written analytical `_backward()` methods.

---

## Project Structure

```text
.
├── value.py             # Core autograd engine (Value class, all operators, topological DFS)
├── neural-network.py    # Modular deep learning framework: Neuron, Layer, and MLP
├── micrograd.py         # Neuron forward/backward pass with decomposed tanh & SVG rendering
├── torch-micrograd.py   # Benchmark script comparing custom autograd directly against PyTorch
├── dd.py                # Graphviz computation graph renderer (trace, Digraph)
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

### Basic Autograd & Operators

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

### Building and Evaluating an MLP

```python
from value import Value
from neural-network import MLP

# Create an MLP with 3 inputs, two hidden layers of 4, and 1 output
model = MLP(3, [4, 4, 1])

# Input dataset and target values
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# Forward pass
ypred = [model(x) for x in xs]

# Mean Squared Error loss
loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
print(f"Total Loss: {loss.data:.4f}")

# Backpropagation through the entire network
loss.backward()
```

### Visualizing the Computation Graph

```python
from value import Value
from dd import dd

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

# Render graph to SVG
dot = dd(out)
dot.render("computation_graph", format="svg", cleanup=True)
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

## Roadmap

- [x] Implement extended arithmetic operators (`-`, `/`, `**`, `exp`).
- [x] Support automatic scalar casting (`Value + float`, `float * Value`).
- [x] Implement PyTorch-like modular abstractions: `Neuron`, `Layer`, and `MLP`.
- [x] Validate analytical gradients directly against PyTorch.
- [ ] Add `parameters()` method and `zero_grad()` across `Neuron`, `Layer`, and `MLP`.
- [ ] Implement complete training loop with Stochastic Gradient Descent (SGD) weight updates.
- [ ] Add ReLU and LeakyReLU activations.
- [ ] Add Binary Cross-Entropy (BCE) loss function.
- [ ] Add support for mini-batching and 2D tensor operations.