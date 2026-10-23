# Neural Networks & Autograd Engine from Scratch

A lightweight, scalar-valued automatic differentiation (autograd) engine and neural network framework built from scratch in Python.

This project implements the foundational mathematics of deep learning—representing mathematical expressions as computational graphs, performing automatic backpropagation via the chain rule using topological sorting, and visually rendering computation graphs.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
  - [1. Scalar Autodiff Engine (`value.py`)](#1-scalar-autodiff-engine-valuepy)
  - [2. Computation Graph Visualization (`dd.py`)](#2-computation-graph-visualization-ddpy)
  - [3. Artificial Neuron Example (`micrograd.py`)](#3-artificial-neuron-example-microgradpy)
  - [4. Multi-Layer Feedforward Network (`nn.py`)](#4-multi-layer-feedforward-network-nnpy)
  - [5. Gradient Verification (`functions.py`)](#5-gradient-verification-functionspy)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Quickstart & Usage](#quickstart--usage)
- [How Backpropagation Works Here](#how-backpropagation-works-here)
- [Roadmap](#roadmap)

---

## Project Overview

Modern deep learning frameworks like PyTorch and TensorFlow abstract away the underlying mechanics of forward passes, computational graphs, and reverse-mode automatic differentiation.

This repository demonstrates how these components function under the hood:
- **Scalar-level tracking**: Tracking values, mathematical operations, and parent dependencies dynamically.
- **Reverse-mode autodiff**: Evaluating local derivatives and propagating gradients backward using the chain rule.
- **Topological DAG traversal**: Ordering graph nodes dynamically so gradients flow correctly without cycle errors or missing dependencies.
- **Visual graph rendering**: Exporting computation graphs with Graphviz to observe values and gradients at every intermediate step.

---

## Key Features

- **`Value` Scalar Engine**: Encapsulates scalar values, tracks operations (`+`, `*`, `tanh`), and accumulates gradients (`grad`).
- **Chain Rule & Backpropagation**: Implements exact analytical derivatives with an automated `.backward()` call.
- **Topological Sort**: Dynamically resolves execution order of the Directed Acyclic Graph (DAG) before backpropagation.
- **Visual DAG Inspection**: Visualizes nodes, operations, intermediate outputs, and gradients in an SVG format using Graphviz.
- **Vectorized Forward Pass**: Layer-by-layer forward propagation demo using NumPy with Sigmoid activations.

---

## Core Components

### 1. Scalar Autodiff Engine (`value.py`)

The `Value` class is the core building block of the autograd engine:
- **`data`**: The raw scalar value (float).
- **`grad`**: The accumulated gradient ($\frac{\partial L}{\partial v}$), initialized to `0.0`.
- **`_prev`**: A set tracking parent `Value` nodes that produced this node.
- **`_op`**: The operation string (e.g. `'+'`, `'*'`, `'tanh'`).
- **`_backward`**: A function closure that computes local gradients and propagates them to parent nodes using the chain rule.

Supported mathematical operations:
- **Addition (`__add__`)**: Local gradients are distributed equally:
  $$\frac{\partial (a + b)}{\partial a} = 1, \quad \frac{\partial (a + b)}{\partial b} = 1$$
- **Multiplication (`__mul__`)**: Product rule:
  $$\frac{\partial (a \cdot b)}{\partial a} = b, \quad \frac{\partial (a \cdot b)}{\partial b} = a$$
- **Hyperbolic Tangent (`tanh`)**: Activation function:
  $$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

#### Reverse-Mode Automatic Differentiation (`backward()`)
When calling `.backward()` on an output node:
1. It builds a **topological ordering** using Depth-First Search (`DFS`) so that every node appears after all its input nodes.
2. It sets the root output node's gradient to `1.0` ($\frac{\partial \text{out}}{\partial \text{out}} = 1.0$).
3. It iterates through the topological list in reverse order, executing each node's `_backward()` function to propagate gradients down to the inputs.

---

### 2. Computation Graph Visualization (`dd.py`)

Visualizes the computational graph created by expressions of `Value` objects using [Graphviz](https://graphviz.org/):
- **`trace(root)`**: Traverses backwards from the root node to collect all reachable nodes and directed edges.
- **`dd(root)`**: Generates a Graphviz `Digraph` layout:
  - Each node shows `{ label | data | grad }`.
  - Intermediate operations (`+`, `*`, `tanh`) are displayed as separate operational nodes connecting inputs to outputs.
  - Generates SVG outputs (such as `draw.svg`).

---

### 3. Artificial Neuron Example (`micrograd.py`)

Demonstrates an artificial neuron with two inputs ($x_1, x_2$), corresponding synaptic weights ($w_1, w_2$), and a bias ($b$):

$$\text{net} = x_1 w_1 + x_2 w_2 + b$$
$$O = \tanh(\text{net})$$

1. Computes the forward pass.
2. Executes `out.backward()` to calculate gradients for all inputs and weights.
3. Renders the graph to `draw.svg` via `dd(out).render("draw")`.

---

### 4. Multi-Layer Feedforward Network (`nn.py`)

Provides a vectorized forward pass of a multi-layer perceptron (MLP) with layer configuration `[2, 3, 3, 1]`:
- Initializes random weight matrices and zero biases.
- Applies matrix multiplication and Sigmoid activation:
  $$a^{(l)} = \sigma(a^{(l-1)} W^{(l)} + b^{(l)})$$

---

### 5. Gradient Verification (`functions.py`)

Implements numerical gradient approximation using finite differences:

$$\frac{df(x)}{dx} \approx \frac{f(x + h) - f(x)}{h}$$

This is used to verify and sanity-check analytical gradients produced by `backward()`.

---

## Project Structure

```text
.
├── value.py          # Core autograd engine (Value class, chain rule, topological sort)
├── dd.py             # Computation graph visualizer using Graphviz (trace and dd)
├── micrograd.py      # Example: 2-input neuron forward & backward pass, SVG render
├── nn.py             # Vectorized feedforward neural network demo using NumPy
├── functions.py      # Finite difference gradient checking and numerical verification
├── equations.py      # Sample mathematical test equations (e.g., parabola)
├── draw.svg          # Rendered computation graph output
├── requirements.txt  # Python package dependencies
└── Readme.md         # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9+
- [Graphviz](https://graphviz.org/download/) (required for graph visualization)
  - **macOS**: `brew install graphviz`
  - **Ubuntu / Debian**: `sudo apt-get install graphviz`
  - **Windows**: `winget install Graphviz` or `choco install graphviz`

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/SankalpPimpalkar/neural-networks.git
cd neural-networks

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install graphviz
```

---

## Quickstart & Usage

### 1. Basic Arithmetic & Backpropagation

```python
from value import Value

# Define scalar variables with labels
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# Build the computation graph: d = (a * b) + c
e = a * b;   e.label = 'e'
d = e + c;   d.label = 'd'
f = Value(-2.0, label='f')
L = d * f;   L.label = 'L'

# Run backpropagation
L.backward()

# Inspect gradients (dL/da, dL/db, etc.)
print(f"L data: {L.data}")   # -8.0
print(f"dL/da:  {a.grad}")   # 6.0
print(f"dL/db:  {b.grad}")   # -4.0
print(f"dL/dc:  {c.grad}")   # -2.0
```

### 2. Visualizing the Computation Graph

```python
from value import Value
from dd import dd

# Build an expression
x1 = Value(2.0, label='x1')
w1 = Value(-3.0, label='w1')
b = Value(6.881, label='b')

n = x1 * w1 + b
n.label = 'n'
o = n.tanh()
o.label = 'out'

# Compute gradients
o.backward()

# Render graph to SVG (saves as 'graph.svg')
dot = dd(o)
dot.render("graph", format="svg", cleanup=True)
```

---

## How Backpropagation Works Here

1. **Local Gradients via Closures**:
   When an operation occurs (e.g. `out = a * b`), `out` stores a custom `_backward` function that knows how to differentiate that specific operation:
   ```python
   def _backward():
       self.grad += other.data * out.grad
       other.grad += self.data * out.grad
   ```
2. **Topological Sort**:
   Because graphs can branch and merge, gradients must be evaluated in reverse dependency order. `build_topo()` performs a recursive depth-first post-order traversal:
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
3. **Gradient Accumulation**:
   Gradients are accumulated with `+=` rather than assigned (`=`) so that if a node is used multiple times in an expression, its gradients correctly sum according to the multivariate chain rule.

---

## Roadmap

- [ ] Implement additional math operators (`-`, `/`, `**`, `exp`, `log`).
- [ ] Add support for operations with non-`Value` scalars (`Value + float`, `float * Value`).
- [ ] Add ReLU and LeakyReLU activation functions.
- [ ] Implement PyTorch-like modular abstractions: `Module`, `Neuron`, `Layer`, and `MLP`.
- [ ] Implement Loss functions (Mean Squared Error, Binary Cross-Entropy).
- [ ] Implement training loop with Stochastic Gradient Descent (SGD).