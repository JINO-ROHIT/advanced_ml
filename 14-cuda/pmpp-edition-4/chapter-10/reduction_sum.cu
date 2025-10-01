// single block multiple threads
#include <common.cuh>

#define BLOCK_DIM 1024

__global__ void simple_sum_reduction_kernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads(); // __syncthreads() ensures all threads complete their additions for the current stride before any thread moves to the next stride. cant put after for loop.
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__global__ void covergent_sum_reduction_kernel_reversed(float* input, float* output) {
    unsigned int i = threadIdx.x + blockDim.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        // stride iterations remains the same, but we just use it to index the previous input to be taken
        if (blockDim.x - threadIdx.x <= stride) {
            input[i] += input[i - stride];
        }
        __syncthreads();
    }
    // take it from the last input
    if (threadIdx.x == blockDim.x - 1) {
        *output = input[i];
    }
}

float simple_parallel_sum_reduction(float* data, int length) {
    assert(length == 2 * BLOCK_DIM && "Length must be equal to 2 * BLOCK_DIM");

    float total;
    float* d_total;
    float* d_data;

    dim3 dimBlock(BLOCK_DIM);  // we always run this with as much threads in block as possible
    dim3 dimGrid(1);           // since the blocks can't communicate we are stuck for now with a single block

    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    simple_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

float covergent_parallel_sum_reduction_reversed(float* data, int length) {
    assert(length == 2 * BLOCK_DIM && "Length must be equal to 2 * BLOCK_DIM");

    float total;
    float* d_total;
    float* d_data;

    dim3 dimBlock(BLOCK_DIM);  // we always run this with as much threads in block as possible
    dim3 dimGrid(1);           // since the blocks can't communicate we are stuck for now with a single block

    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    covergent_sum_reduction_kernel_reversed<<<dimGrid, dimBlock>>>(d_data, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}