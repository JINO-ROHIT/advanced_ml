#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int M = 1000;
const int N = 1000;
const int THREADS_PER_BLOCK = 256;

__global__ void add_ten(const float *a, float *out, int row, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < row * col) {
        out[idx] = a[idx] + 10;
    }
}

int main() {
    float *h_A, *h_out;
    float *d_A, *d_out;
    
    h_A = new float[M * N];
    h_out = new float[M * N];
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = rand() / (float)RAND_MAX;
        }
    }
    
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_out, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    int numBlocks = (M * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_ten<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_out, M, N);
    
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < min(5, M); i++) {
        for (int j = 0; j < min(5, N); j++) {
            printf("h_out[%d][%d] = %f\n", i, j, h_out[i * N + j]);
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_out);
    delete[] h_A;
    delete[] h_out;
}