### Core Definitions

- **Free Indices** - The indices specified in the output.
- **Summation Indices** - All the other indices. Those are appear in input but NOT in output.

```
M = np.einsum('ik, kj -> ij', A, B)

free indices - i, j
summation index - k
```


### Main Rules

- Repeating letter in the inputs means those values will be multiplied and those product will be the output.

```
M = np.einsum('ik, kj -> ij', A, B)
```

- Omitting a letter means that axis will be summed.

```
X = np.ones(3)
M = np.einsum('i ->', X)
```

- We can return the unsummed axes in any order.

```
X = np.ones((5, 10, 15))
M = np.einsum('ijk -> kji', X)
```