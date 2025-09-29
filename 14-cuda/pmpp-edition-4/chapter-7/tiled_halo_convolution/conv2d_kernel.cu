#define DEFINE_CONSTANT_MEMORY
#include <stdio.h>

#include "conv2d_kernel.cuh"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

cudaError_t initConstFilter(const float* filter, int r) {
    cudaError_t err;
    // First, get the symbol address to ensure it's valid
    void* d_ptr;
    size_t size;
    
    err = cudaGetSymbolAddress(&d_ptr, constFilter);
    if (err != cudaSuccess) {
        printf("Error getting symbol address: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    err = cudaGetSymbolSize(&size, constFilter);
    if (err != cudaSuccess) {
        printf("Error getting symbol size: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    size_t expected_size = (2 * r + 1) * (2 * r + 1) * sizeof(float);
    if (size < expected_size) {
        printf("Error: Not enough constant memory allocated (have %zu bytes, need %zu bytes)\n", 
               size, expected_size);
        return cudaErrorInvalidValue;
    }
    
    err = cudaMemcpyToSymbol(constFilter, filter, expected_size, 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    }
    return err;
}


__global__ void tiled_convolution_kernel(float* M, float* P, int r, int height, int width) {
    int row = blockIdx.y * OUT_TILE_SIZE + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUT_TILE_SIZE + threadIdx.x - FILTER_RADIUS;

    __shared__ float M_s[IN_TILE_SIZE][IN_TILE_SIZE];

    if (row >= 0 && row < height && col >= 0 && col < width) {
        M_s[threadIdx.y][threadIdx.x] = M[row * width + col];
    } else {
        M_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int filterIndex;

    if (row >= 0 && row < height && col >= 0 && col < width) {
        if (tileRow >= 0 && tileRow < OUT_TILE_SIZE && tileCol >= 0 && tileCol < OUT_TILE_SIZE) {
            float Pvalue = 0.0f;

            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    filterIndex = fRow * (2 * FILTER_RADIUS + 1) + fCol;
                    Pvalue += M_s[threadIdx.y + fRow - FILTER_RADIUS][threadIdx.x + fCol - FILTER_RADIUS] *
                              constFilter[filterIndex];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor conv2d_torch_with_tiled_convolution(torch::Tensor input, torch::Tensor kernel, int r) {
    TORCH_CHECK(input.device().type() == torch::kCUDA, "Input must be a CUDA tensor");
    TORCH_CHECK(kernel.device().type() == torch::kCUDA, "Kernel must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "Kernel must be float32");

    int height = input.size(0);
    int width = input.size(1);
    auto output = torch::empty_like(input);
    cudaError_t error;

    error = initConstFilter(kernel.data_ptr<float>(), r);
    if (error != cudaSuccess) {
        std::cout << "Failed to initialize constant memory: " << cudaGetErrorString(error) << std::endl;
        return output;
    }

    const dim3 dimBlock(IN_TILE_SIZE, IN_TILE_SIZE);
    dim3 dimGrid(cdiv(width + 2 * FILTER_RADIUS, OUT_TILE_SIZE), cdiv(height + 2 * FILTER_RADIUS, OUT_TILE_SIZE));

    tiled_convolution_kernel<<<dimGrid, dimBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), r, height, width);

    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_torch", &conv2d_torch_with_tiled_convolution,
          "2D Convolution with CUDA utilizing constant memory");
}