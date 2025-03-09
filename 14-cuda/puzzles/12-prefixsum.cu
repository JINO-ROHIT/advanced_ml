//  Implement a kernel that computes a sum over a and stores it in out. If the size of a is greater than the block size, only store the sum of each block.
// We will do this using the parallel prefix sum algorithm in shared memory. That is, each step of the algorithm should sum together half the remaining numbers.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000; 
const int THREADS_PER_BLOCK = 256; 

__global__ void prefixSum(const float *a, float *out, int ds) {
    __shared__ float s_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    s_data[tid] = (index < ds) ? a[index] : 0.0f;
    __syncthreads();

    // Perform reduction to compute sum within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    // Store the block sum in output
    if (tid == 0) {
        out[blockIdx.x] = s_data[0];
    }
}

int main() {
    float *h_A, *h_out;
    float *d_A, *d_out;

    int numBlocks = (DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    h_A = new float[DSIZE];
    h_out = new float[numBlocks];

    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float) RAND_MAX;
    }


    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_out, numBlocks * sizeof(float));


    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);


    prefixSum<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_out, DSIZE);
    cudaDeviceSynchronize();


    cudaMemcpy(h_out, d_out, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0; i < numBlocks; i++) {
        printf("Block %d sum = %f\n", i, h_out[i]);
    }


    cudaFree(d_A);
    cudaFree(d_out);
    delete[] h_A;
    delete[] h_out;

    return 0;
}
