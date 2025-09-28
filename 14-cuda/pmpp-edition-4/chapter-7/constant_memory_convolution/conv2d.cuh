#include <cuda_runtime.h>

#define FILTER_RADIUS 9
#define BLOCK_SIZE 32

// Define the constant memory in the header with extern linkage
#ifdef __CUDACC__
#define CONSTANT __constant__
#else
#define CONSTANT extern __constant__
#endif

CONSTANT float constFilter[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];

__global__ void conv2d_kernel_const(float* M, float* P, int r, int width, int height);

cudaError_t initConstFilter(const float* filter, int r);