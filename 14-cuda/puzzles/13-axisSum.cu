//  Implement a kernel that computes a sum over each column of a and stores it in out.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000; 
const int THREADS_PER_BLOCK = 256; 

__global__ void axisSum(const float *a, float *out, int ds) {
    __shared__ float sdata[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    while (i < ds) {
        sum += a[i];
        i += gridDim.x * blockDim.x; 
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
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


    axisSum<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_out, DSIZE);
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
