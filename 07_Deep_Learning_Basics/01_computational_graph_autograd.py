import numpy as np


class Value:
    """Scalar computational node supporting forward operations and automatic reverse-mode autodiff."""

    def __init__(self, data: float, _children=(), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Topological sort order for backpropagation."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    # Computation: L = ReLU(w * x + b)
    x = Value(2.0)
    w = Value(-3.0)
    b = Value(8.0)

    # Forward pass: (-3.0 * 2.0) + 8.0 = 2.0 -> ReLU(2.0) = 2.0
    u = w * x
    v = u + b
    L = v.relu()

    # Backward pass (Chain Rule across the computational graph)
    L.backward()

    print("--- Computational Graph & Autograd Verification ---")
    print(f"Forward Output L: {L.data:.4f}")
    print(f"dL/dw (Expected: x = 2.0): {w.grad:.4f}")
    print(f"dL/dx (Expected: w = -3.0): {x.grad:.4f}")
    print(f"dL/db (Expected: 1.0):      {b.grad:.4f}")