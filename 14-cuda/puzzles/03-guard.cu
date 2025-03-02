// Implement a kernel that adds 10 to each position of a and stores it in out. You have more threads than positions. (dont use a block, since use many threads)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000; 
const int THREADS_COUNT = 1024;

__global__ void add_ten(const float *a, float *out, int ds){
    int idx = threadIdx.x;

    if (idx < ds) {
        out[idx] = a[idx] + 10.0f;
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

    add_ten<<<1, THREADS_COUNT>>>(d_A, d_out, DSIZE);

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
