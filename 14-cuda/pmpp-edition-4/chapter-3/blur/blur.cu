#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h, int channels, int BLUR_SIZE) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {

        for (int c = 0; c < channels; ++c) {
            int pixVal = 0;
            int pixels = 0;

            // Average of surrounding blur_size x blur_size box
            for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {

                        pixVal += in[(curRow * w + curCol) * channels + c];
                        ++pixels;
                    }
                }
            }

            out[(row * w + col) * channels + c] = (unsigned char)(pixVal / pixels);
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor gaussian_blur(torch::Tensor img, int blurSize) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);
    const auto channels = img.size(2);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width, channels}, 
                              torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    blurKernel<<<dimGrid, dimBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(), 
        result.data_ptr<unsigned char>(), 
        width, height, channels, blurSize);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}