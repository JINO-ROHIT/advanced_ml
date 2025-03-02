// Implement a kernel that adds a and b and stores it in out. Inputs a and b are vectors. You have more threads than positions.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int SIZE = 1000;
const int THREADS_PER_DIM = 32;

__global__ void broadcast(const float *a, const float *b, float *out, int size){
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;

    int curr_idx = idx_x * size + idx_y;

    if(idx_x < size && idx_y < size){
        out[curr_idx] = a[idx_x] + b[idx_y];
    }
}

int main() {
    float *h_A, *h_B, *h_OUT;
    float *d_A, *d_B, *d_OUT;

    h_A = new float[SIZE * 1];
    h_B = new float[1 * SIZE];
    h_OUT = new float[SIZE * SIZE];

    for (int i = 0; i < SIZE; i++){
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc(&d_A, sizeof(float) * SIZE * 1);
    cudaMalloc(&d_B, sizeof(float) * 1 * SIZE);
    cudaMalloc(&d_OUT, sizeof(float) * SIZE * SIZE);

    cudaMemcpy(d_A, h_A, sizeof(float) * SIZE * 1, cudaMemcpyHostToDevice);

    dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);
    broadcast<<<1, threads>>>(d_A, d_B, d_OUT, SIZE);

    cudaMemcpy(h_OUT, d_OUT, sizeof(float) * SIZE * SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < min(5, SIZE); i++) {
        for (int j = 0; j < min(5, SIZE); j++) {
            printf("h_out[%d][%d] = %f\n", i, j, h_OUT[i * SIZE + j]);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_OUT);

    delete[] h_A;
    delete[] h_B;
    delete[] h_OUT;
}