#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000; 
const int THREADS_PER_BLOCK = 16; 

__global__ void blocks2d(const float *a, float *out, int ds) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ds && col < ds) {
        int idx = row * ds + col;
        out[idx] = a[idx] + 10.0f;
    }
}

int main() {
    float *h_A, *h_out;
    float *d_A, *d_out;

    h_A = new float[DSIZE * DSIZE];
    h_out = new float[DSIZE * DSIZE];

    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_out, DSIZE * DSIZE * sizeof(float));

    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks((DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 
                   (DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    blocks2d<<<numBlocks, threadsPerBlock>>>(d_A, d_out, DSIZE);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("h_out[%d][%d] = %f\n", i, j, h_out[i * DSIZE + j]);
        }
    }

    cudaFree(d_A);
    cudaFree(d_out);
    delete[] h_A;
    delete[] h_out;

    return 0;
}
