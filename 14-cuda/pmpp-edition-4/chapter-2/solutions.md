1. If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?

`i = blockIdx.x * blockDim.x + threadIdx.x`

2. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

`i = (blockIdx.x * blockDim.x + threadIdx.x) * 2`

4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

Each block has 1024 threads, we need to have 8 thread blocks to process 8000 elements. `8 * 1024 = 8192` threads.  ceil(800 / 1024)


9. Consider the following CUDA kernel and the corresponding host function
that calls it:

```c
01 __global__ void foo_kernel(float* a, float* b, unsigned int N) {
02     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
03     
04     if (i < N) {
05         b[i] = 2.7f * a[i] - 4.3f;
06     }
07 }
08 
09 void foo(float* a_d, float* b_d) {
10     unsigned int N = 200000;
11     foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
12 }
```

**a. What is the number of threads per block?**
`128`

**b. What is the number of threads in the grid?**
`1563 * 128 = 200064`


**c. What is the number of blocks in the grid?**
`1563`

**d. What is the number of threads that execute the code on line 02?**
`1563 * 128 = 200064`

**e. What is the number of threads that execute the code on line 04?**
`N = 200000`
