// Implement a kernel that multiplies square matrices a and b and stores the result in out.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000;
const int THREADS_PER_BLOCK = 16; 

__global__ void matrixMul(const float *a, const float *b, float *out, int ds) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < ds && col < ds) {
        float sum = 0.0f;
        for (int k = 0; k < ds; k++) {
            sum += a[row * ds + k] * b[k * ds + col];
        }
        out[row * ds + col] = sum;
    }
}

int main() {
    float *h_A, *h_B, *h_out;
    float *d_A, *d_B, *d_out;


    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_out = new float[DSIZE * DSIZE];


    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }


    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_out, DSIZE * DSIZE * sizeof(float));


    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks((DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  (DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);


    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_out, DSIZE);
    cudaDeviceSynchronize();


    cudaMemcpy(h_out, d_out, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);


    printf("Sample results:\n");
    for (int i = 0; i < 5; i++) {
        printf("out[0][%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
    delete[] h_A;
    delete[] h_B;
    delete[] h_out;

    return 0;
}
