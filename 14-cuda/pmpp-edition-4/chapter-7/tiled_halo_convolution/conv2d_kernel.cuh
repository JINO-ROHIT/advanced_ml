#ifndef CONV2D_KERNEL_CUH
#define CONV2D_KERNEL_CUH

#include <cuda_runtime.h>

#define FILTER_RADIUS 9
#define BLOCK_SIZE 32
#define TILE_SIZE 32

#define IN_TILE_SIZE 32
#define OUT_TILE_SIZE (IN_TILE_SIZE - 2 * FILTER_RADIUS)

// Use a macro to control where constant memory is defined
#ifdef DEFINE_CONSTANT_MEMORY
__constant__ float constFilter[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];
#else
extern __constant__ float constFilter[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];
#endif


__global__ void tiled_convolution_kernel(float* M, float* P, int r, int height, int width);
cudaError_t initConstFilter(const float* filter, int r);

#endif