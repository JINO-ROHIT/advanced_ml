#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 4096;
const int block_size = 256;

__global__ void vadd(const float *A, const float *B, float *C, int ds) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ds) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_c;

    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    // Initialize host arrays
    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float) RAND_MAX;
        h_B[i] = rand() / (float) RAND_MAX;
        h_C[i] = 0;
    }

    // Allocate memory on GPU
    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * sizeof(float));
    cudaMalloc(&d_c, DSIZE * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int grid_size = (DSIZE + block_size - 1) / block_size;
    vadd<<<grid_size, block_size>>>(d_A, d_B, d_c, DSIZE);

    // Ensure kernel execution is finished before copying back
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_c, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);  // Should print A[0] + B[0]

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_c);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
