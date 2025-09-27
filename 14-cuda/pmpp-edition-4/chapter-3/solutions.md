3. Consider the following CUDA kernel and the corresponding host function that calls it:

```cu
01 __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
02     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
03     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
04     if (row < M && col < N) {
05         b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
06     }
07 }
08 void foo(float* a_d, float* b_d) {
09     unsigned int M = 150;
10     unsigned int N = 300;
11     dim3 bd(16, 32);
12     dim3 gd((N - 1) / 16 + 1, ((M - 1) / 32 + 1));
13     foo_kernel <<< gd, bd >>> (a_d, b_d, M, N);
14 }
```

**a. What is the number of threads per block?**

The number of threads in a single block can be inferred from the variable `bd` (blockDim), it is `16 x 32 = 512` threads. 


**b. What is the number of threads in the grid?**

Total number of threads in the grid is `Number of blocks in the grid×Number of threads per block` so in this case `512 x 95 = 48,640` (see  **3a** and **3c**).

**c What is the number of blocks in the grid?**

The number of blocks in a grid can be inferred from the variable `gd` (gridDim). It is `((N - 1) / 16 + 1, ((M - 1) / 32 + 1));` where `M=150` and `N=300`. Hence `((300-1)/16 + 1, (150-1)/32 + 1)` -> `(299/16 + 1, 149/32 + 1)` -> `(18  + 1, 4  + 1)` -> `(19, 5)` -> `19 x 5 = 95` blocks. 

**d. What is the number of threads that execute the code on line 05?**

Block Dim = `(16, 32) = 512 threads`
Grid Dim = `(19, 5) = 95 blocks`

Total threads = `95 * 512 = 48,640.`

But only threads with `0 <= row < 150` and `0 <= col < 300`
Basically `M x N` threads = `45,000` threads.

4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:
- **a.** If the matrix is stored in row-major order.

In the row-major order the way the array is linearized using the formula `row x width + col`, so the index will be `20 x 400 + 10 = 8,010`.

- **b.** If the matrix is stored in column-major order.

In the column-major order the array is linearized using the formula `col x height + row`, so the index will be `10 x 500 + 20 = 5,020`

5. Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5.

The linearized index of of an element in a 3d tensor will be calculated using the foltmula `plane x width x height + row x width + col`, so the index will be `5 x 400 x 500 + 20 x 400 + 10 = 100,000,000 + 8,000 + 10 = 1.008.010`