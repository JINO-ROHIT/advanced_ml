// Implement a kernel that adds 10 to each position of a and stores it in out. Input a is 2D and square. You have more threads than positions.(again dont use a block)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int M = 1000;
const int N = 1000;
const int THREADS_PER_DIM = 32;

__global__ void add_ten(const float *a, float *out, int row, int col) {
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;

    int curr_idx = idx_x * row + idx_y;

    if (idx_x < row && idx_y < col) {
        out[curr_idx] = a[curr_idx] + 10;
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

    dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);
    add_ten<<<1, threads>>>(d_A, d_out, M, N);
    
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