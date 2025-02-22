#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define the size of the arrays
int DSIZE = 1024;

// CUDA kernel function to add elements of two arrays
__global__ void zip(float *A, float *B, float *C, int ds) {
    // Calculate the thread index
    int idx = threadIdx.x;
    // Perform the addition if the index is within bounds
    if (idx < ds) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Host pointers for arrays
    float *h_A, *h_B, *h_C;
    // Device pointers for arrays
    float *d_A, *d_B, *d_C;

    // Allocate memory on the host
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    // Initialize host arrays with random values
    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory on the device
    cudaMalloc(&d_A, sizeof(float) * DSIZE);
    cudaMalloc(&d_B, sizeof(float) * DSIZE);
    cudaMalloc(&d_C, sizeof(float) * DSIZE);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeof(float) * DSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * DSIZE, cudaMemcpyHostToDevice);

    // Launch the kernel with one block of DSIZE threads
    zip<<<1, DSIZE>>>(d_A, d_B, d_C, DSIZE);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, sizeof(float) * DSIZE, cudaMemcpyDeviceToHost);

    // Print the first 10 elements of the result
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f \n", i, h_C[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}
