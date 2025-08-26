import torch

x = torch.rand((2, 3))

# permutation
op = torch.einsum("ij -> ji", x)
print(op)

# summation
op = torch.einsum("ij -> ", x)
print(op)

# column sum
op = torch.einsum("ij -> j", x)
print(op)

# row sum
op = torch.einsum("ij -> i", x)
print(op)

# matrix vector multiplication
v = torch.ones((1, 3))
op = torch.einsum("ij, kj -> ik", x, v)
print(op)

# matrix matrix multiplication
op = torch.einsum("ij, kj -> ik", x, x)
print(op)

# dot product with matrix
op = torch.einsum("ij, kj -> ", x, x)
print(op)

# element wise multiplication
op = torch.einsum("ij, ij -> ij", x, x)
print(op)