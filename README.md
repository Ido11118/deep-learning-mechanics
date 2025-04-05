# Sampling and Gradients in PyTorch

This project implements two core deep learning components in PyTorch **from scratch**:
1. Sampling from a discrete probability distribution
2. Scalar-based automatic differentiation with backpropagation

It was developed as part of an academic assignment to explore and replicate internal mechanisms found in deep learning frameworks like PyTorch.

---

## 📁 Implemented Components

### 🎲 Discrete Sampler

- `my_sampler(size, dist)`  
  Samples values from a given discrete probability distribution using **inverse transform sampling**.

- A histogram visualization is included to confirm that sampled frequencies align with the expected probabilities.

### 🔁 Scalar Autograd Engine

- `MyScalar`  
  Represents a scalar value in a computation graph. Stores its value, parents, and local derivatives.

- `get_gradient(output)`  
  Performs recursive backpropagation through the computation graph to compute gradients w.r.t. inputs.

- Supported operations:
  - Unary: `power`, `exp`, `ln`, `sin`, `cos`
  - Binary: `add`, `mul`, `div`
  - Constants: `add_const`, `mul_const`

---

## ✅ Constraints

- Sampling uses **no built-in functions** like `torch.multinomial` or `random.choices`.
- Autograd implementation avoids `.backward()`, `.grad`, or any PyTorch autograd internals.
- Operations work on scalar values only (not tensors).

---

## 🤐 Educational Purpose

This project helped us gain a deeper understanding of two foundational topics in deep learning:
- How values are sampled from custom distributions using cumulative probability logic.
- How gradients flow through a computation graph using the chain rule and backpropagation.

By recreating these mechanisms manually, we gained better intuition for how high-level libraries work behind the scenes.

---

## 👤 Author

Ido  
GitHub: [@Ido11118](https://github.com/Ido11118)
