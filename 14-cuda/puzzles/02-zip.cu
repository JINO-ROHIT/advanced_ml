// Implement a kernel that adds together each position of a and b and stores it in out. You have 1 thread per position.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int DSIZE = 1024;

__global__ void zip(float *A, float *B, float *C, int ds) {
    int idx = threadIdx.x;
    if (idx < ds) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc(&d_A, sizeof(float) * DSIZE);
    cudaMalloc(&d_B, sizeof(float) * DSIZE);
    cudaMalloc(&d_C, sizeof(float) * DSIZE);

    cudaMemcpy(d_A, h_A, sizeof(float) * DSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * DSIZE, cudaMemcpyHostToDevice);

    zip<<<1, DSIZE>>>(d_A, d_B, d_C, DSIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeof(float) * DSIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f \n", i, h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}
