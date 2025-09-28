#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

__global__ void conv2d_kernel(float *N, float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outCol >= width || outRow >= height) return;

    float Psum = 0.0f;
    // left and right of the centre point + the centre point itself
    int filterWidth = 2 * r + 1; 

    for (int fRow = 0; fRow < filterWidth; fRow++) {
        for (int fCol = 0; fCol < filterWidth; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;

            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Psum += F[fRow * filterWidth + fCol] * N[inRow * width + inCol];
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

    const dim3 dimBlock(32, 32);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    conv2d_kernel<<<dimGrid, dimBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(), r, height, width);

    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_torch", &conv2d_torch, "Custom 2D convolution (CUDA)");
}