// Implement a kernel that computes a 1D convolution between a and b and stores it in out

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000; 
const int THREADS_PER_BLOCK = 256; 
const int KERNEL_SIZE = 3;

__constant__ float d_kernel[KERNEL_SIZE];

__global__ void Conv1D(const float *a, float *out, int ds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= ds) return;

    float sum = 0.0f;

    for (int i = 0; i < KERNEL_SIZE; i++) {
        int ai = idx + i - KERNEL_SIZE / 2;

        if (ai >= 0 && ai < ds) {
            sum += a[ai] * d_kernel[i];
        }
    }

    out[idx] = sum;
}

int main(){
    float *h_A, *h_out;
    float *d_A, *d_out;

    h_A = new float[DSIZE];
    h_out = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float) RAND_MAX;
    }


    float h_kernel[KERNEL_SIZE] = {0.2f, 0.5f, 0.2f}; 
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * sizeof(float));


    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_out, DSIZE * sizeof(float));


    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);


    int numBlocks = (DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    Conv1D<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_out, DSIZE);

    cudaDeviceSynchronize();


    cudaMemcpy(h_out, d_out, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }


    cudaFree(d_A);
    cudaFree(d_out);
    delete[] h_A;
    delete[] h_out;

    return 0;
}
