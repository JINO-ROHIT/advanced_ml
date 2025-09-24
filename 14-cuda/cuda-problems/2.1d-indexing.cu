// multiple blocks, 512 threads
// 1d indexing to element wise add.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int DSIZE = 512;
int THREADS_PER_BLOCK = 11;

__global__ void add(float *A, float *B, float *C, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < size){
        C[idx] = A[idx] + B[idx];
        printf("Block: %d, Thread: %d, Global idx: %d\n", 
               blockIdx.x, threadIdx.x, idx);
    }
}

int main() {
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];
    
    for (int i = 0; i < DSIZE; i++) {
        // easy numbers to check addition result
        h_A[i] = i; 
        h_B[i] = i;
    }
    
    cudaMalloc(&d_A, sizeof(float) * DSIZE);
    cudaMalloc(&d_B, sizeof(float) * DSIZE);
    cudaMalloc(&d_C, sizeof(float) * DSIZE);
    
    cudaMemcpy(d_A, h_A, sizeof(float) * DSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * DSIZE, cudaMemcpyHostToDevice);
    
    int num_blocks = (DSIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add<<<num_blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, DSIZE);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, sizeof(float) * DSIZE, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < DSIZE; i++) {
        printf("C[%d] = %f \n", i, h_C[i]);
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
}