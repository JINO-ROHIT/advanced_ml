// Implement a kernel that sums together the last 3 position of a and stores it in out. You have 1 thread per position.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000; 
const int THREADS_PER_BLOCK = 256; 

__global__ void pooling(const float *a, float *out, int ds) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx <= ds - 3) { 
        out[idx] = a[idx] + a[idx + 1] + a[idx + 2];
    } else if (idx < ds) {
        out[idx] = 0;
    }
}

int main(){
    float *h_A, *h_out;
    float *d_A, *d_out;


    h_A = new float[DSIZE];
    h_out = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float) RAND_MAX; 
    }


    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_out, DSIZE * sizeof(float));

    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);


    int numBlocks = (DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    pooling<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_out, DSIZE);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_A);
    cudaFree(d_out);
    delete[] h_A;
}