#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000; 
const int THREADS_PER_BLOCK = 256; 

__global__ void dotproduct(const float *a, const float *b, float *out, int ds) {
    __shared__ float s[THREADS_PER_BLOCK];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (idx < ds) {
        s[tid] = a[idx] * b[idx];
    } else {
        s[tid] = 0.0f;
    }
    __syncthreads();

    // Let thread 0 of each block compute the sum for the block
    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < blockDim.x; k++) {
            sum += s[k];
        }
        out[blockIdx.x] = sum; // Store partial sum in global memory
    }
}

int main() {
    float *h_A, *h_B, *h_partial_sums;
    float *d_A, *d_B, *d_partial_sums;

    int numBlocks = (DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_partial_sums = new float[numBlocks];

    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float) RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * sizeof(float));
    cudaMalloc(&d_partial_sums, numBlocks * sizeof(float));


    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);


    dotproduct<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_partial_sums, DSIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(h_partial_sums, d_partial_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float final_dot_product = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        final_dot_product += h_partial_sums[i];
    }

    printf("Dot Product: %f\n", final_dot_product);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial_sums);
    delete[] h_A;
    delete[] h_B;
    delete[] h_partial_sums;

    return 0;
}
