#include <iostream>

#include "stencil.h"

// Global coefficient variables definition
int c0 = 0;
int c1 = 1;
int c2 = 1;
int c3 = 1;
int c4 = 1;
int c5 = 1;
int c6 = 1;


__global__ void stencil_kernel(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4, int c5,
                               int c6) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] + 
                                     c1 * in[i * N * N + j * N + (k - 1)] +
                                     c2 * in[i * N * N + j * N + (k + 1)] + 
                                     c3 * in[i * N * N + (j - 1) * N + k] +
                                     c4 * in[i * N * N + (j + 1) * N + k] + 
                                     c5 * in[(i - 1) * N * N + j * N + k] +
                                     c6 * in[(i + 1) * N * N + j * N + k];
    }
}

void stencil_3d_parallel_basic(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4, int c5,
                               int c6) {
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL);
    dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(N, dimBlock.y), cdiv(N, dimBlock.z));

    stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();

    error = cudaMemcpy(out, d_out, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to host failed: " << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}