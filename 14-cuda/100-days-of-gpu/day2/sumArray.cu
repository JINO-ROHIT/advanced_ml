#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int DSIZE = 4;

__global__ void sumArray(float *A, float *B, float *C, int ds) {
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

    h_A[0] = 1.0; h_A[1] = 2.0; h_A[2] = 3.0; h_A[3] = 4.0;
    h_B[0] = 5.0; h_B[1] = 6.0; h_B[2] = 7.0; h_B[3] = 8.0;

    cudaMalloc(&d_A, sizeof(float) * DSIZE);
    cudaMalloc(&d_B, sizeof(float) * DSIZE);
    cudaMalloc(&d_C, sizeof(float) * DSIZE);

    cudaMemcpy(d_A, h_A, sizeof(float) * DSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * DSIZE, cudaMemcpyHostToDevice);

    sumArray<<<1, 32>>>(d_A, d_B, d_C, DSIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeof(float) * DSIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < DSIZE; i++) {
        printf("C[%d] = %f \n", i, h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}
