# Manual Broadcasting in PyTorch

This project implements manual tensor broadcasting in PyTorch **without using any built-in broadcasting functions** such as `expand`, `expand_as`, `repeat`, or `torch.broadcast_tensors`.

It was developed as part of an academic assignment focused on understanding and reproducing the broadcasting behavior of PyTorch tensors from scratch.

---

## 📁 Implemented Functions

- `my_expand_as(A, B)`  
  Manually expands tensor `A` to match the shape of tensor `B` using only allowed tensor operations.

- `is_broadcastable(A, B)`  
  Checks whether two tensors can be broadcast together and returns a boolean and the resulting shape.

- `my_broadcast(A, B)`  
  Manually broadcasts both tensors to their common shape (equivalent to `torch.broadcast_tensors(A, B)`).

---

## ✅ Constraints

The implementation avoids all built-in broadcasting helpers, including:
- `expand`, `expand_as`, `repeat`
- `broadcast_to`, `broadcast_tensors`

Instead, it uses only allowed operations such as:
- `unsqueeze`, `stack`, `select`, `reshape`, `clone`

---

## 🚀 Tests

A full test suite is included to validate:
- Correct behavior of manual broadcasting functions
- Consistency with PyTorch’s native broadcasting behavior
- Proper error handling in incompatible cases

---

## 📄 Example

```python
A = torch.tensor([[1], [2]])        # shape: [2,1]
B = torch.tensor([[10, 20, 30]])    # shape: [1,3]

A_b, B_b = my_broadcast(A, B)
print(A_b.shape)  # torch.Size([2, 3])
print(B_b.shape)  # torch.Size([2, 3])
```

---

## 🧰 Educational Purpose

This exercise deepened our understanding of how broadcasting works under the hood and how to manually manipulate tensor shapes using operations like `unsqueeze`, `stack`, and `select`.

---

## 👤 Author

Ido  
GitHub: @Ido11118

