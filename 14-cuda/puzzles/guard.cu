#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000;  // Number of elements
const int THREADS_PER_BLOCK = 256;  // Number of threads per block

__global__ void add_ten(const float *a, float *out, int ds){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread index

    if (idx < ds) {  // Prevent out-of-bounds access
        out[idx] = a[idx] + 10.0f;
    }
}

int main(){
    float *h_A, *h_out;
    float *d_A, *d_out;

    // Allocate memory on host
    h_A = new float[DSIZE];
    h_out = new float[DSIZE];

    // Initialize input data
    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float) RAND_MAX;  // Random values in [0,1]
    }

    // Allocate memory on GPU
    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_out, DSIZE * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with more threads than needed
    int numBlocks = (DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_ten<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_out, DSIZE);

    // Synchronize and copy results back
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_out);
    delete[] h_A;
    delete[] h_out;

    return 0;
}
