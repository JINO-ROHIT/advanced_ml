#include <cuda_runtime.h>
#include <stdio.h>

#define BIN_SIZE 4
#define NUM_BINS ((26 + BIN_SIZE - 1) / BIN_SIZE)

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                \
        cudaError_t error = call;                                                                       \
        if (error != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));\
            exit(EXIT_FAILURE);                                                                         \
        }                                                                                               \
    } while (0)

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void histo_kernel(char* data, unsigned int length, unsigned int* histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd((unsigned int*)&histo[alphabet_position / BIN_SIZE], 1);
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void histogram_parallel(char* data, unsigned int length, unsigned int* histo) {
    char* d_data;
    unsigned int* d_histo;

    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));

    dim3 dimBlock(1024);
    dim3 dimGrid(cdiv(length, dimBlock.x));

    histo_kernel<<<dimGrid, dimBlock>>>(d_data, length, d_histo);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(histo, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histo));
}

int main() {
    const char data[] = "abcdefghijklmnopqrstuvwxyz";
    unsigned int histo[NUM_BINS] = {0};
    histogram_parallel((char*)data, sizeof(data) - 1, histo);

    printf("Histogram bins:\n");
    for (int i = 0; i < NUM_BINS; ++i) {
        printf("Bin %d: %u\n", i, histo[i]);
    }
    return 0;
}