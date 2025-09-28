#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "conv2d.cuh"

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

__global__ void conv2d_kernel_const(float *N, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outCol >= width || outRow >= height) return;

    float Psum = 0.0f;
    // Left and right of the center point + the center point itself
    int filterWidth = 2 * r + 1;

    for (int fRow = 0; fRow < filterWidth; fRow++) {
        for (int fCol = 0; fCol < filterWidth; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;

            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                // Use constant memory F instead of global memory
                Psum += constFilter[fRow * filterWidth + fCol] * N[inRow * width + inCol];
            }
        }
    }

    P[outRow * width + outCol] = Psum;
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor conv2d_torch(torch::Tensor input, torch::Tensor kernel, int r) {
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
    
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    conv2d_kernel_const<<<dimGrid, dimBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), r, width, height);

    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_torch", &conv2d_torch,
          "2D Convolution with CUDA utilizing constant memory");
}