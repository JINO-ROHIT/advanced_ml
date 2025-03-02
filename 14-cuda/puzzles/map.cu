// Implement a "kernel" (GPU function) that adds 10 to each position of vector a and stores it in vector out. You have 1 thread per position.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1024;

__global__ void map(float *a, int ds){
    int idx = threadIdx.x;
    if (idx < ds){
        a[idx] += 10;
    }
}

int main(){
    float *h_A;
    float *d_A;

    h_A = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++)
    {
        h_A[i] = rand() / (float) RAND_MAX;
    }


    cudaMalloc(&d_A, DSIZE * sizeof(float));

    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    map<<<1, DSIZE>>>(d_A, DSIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(h_A, d_A, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("h_A[%d] = %f\n", i, h_A[i]);
    }

    cudaFree(d_A);

    delete[] h_A;
}
