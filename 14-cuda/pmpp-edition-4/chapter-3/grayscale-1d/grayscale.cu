// see the problem with 1d indexing? 479 blocks?

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__global__ void grayscale1d(unsigned char* Pin, unsigned char* Pout, int width, int height){
    int cur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int CHANNELS = 3;

    if(cur_idx < width * height){
        int rgboffset = cur_idx * 3;
        unsigned char r = Pin[rgboffset];
        unsigned char g = Pin[rgboffset + 1];
        unsigned char b = Pin[rgboffset + 2];

        Pout[cur_idx] = 0.21 * r + 0.71 * g + 0.07 * b;
    }
}


inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor rgb_to_gray(torch::Tensor img) {
    TORCH_CHECK(img.device().type() == torch::kCUDA);
    TORCH_CHECK(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);

    dim3 dimBlock(1024);
    dim3 dimGrid(479);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    grayscale1d<<<dimGrid, dimBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(), result.data_ptr<unsigned char>(), width, height);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}