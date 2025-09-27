#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#define TILE_WIDTH 32

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

void clear_l2() {
    // Get actual L2 size via CUDA on first call of this function
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2;  // just to be extra safe (cache is not necessarily strict LRU)
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    // Clear L2 cache (this is run on every call unlike the above code)
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

__global__ void MatrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < o) {
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += M[row * n + i] * N[i * o + col];
        }
        P[row * o + col] = sum;
    }
}


// __global__ void TiledMatrixMulKernel(float* M, float* N, float* P,
//                                      int m, int n, int o) {

//     __shared__ float Mtile[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float Ntile[TILE_WIDTH][TILE_WIDTH];


//     int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
//     int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

//     float value = 0.0f;


//     for (int phase = 0; phase < (n + TILE_WIDTH - 1) / TILE_WIDTH; phase++) {

//         // tile M
//         int mCol = phase * TILE_WIDTH + threadIdx.x;
//         if (row < m && mCol < n)
//             Mtile[threadIdx.y][threadIdx.x] = M[row * n + mCol];
//         else
//             Mtile[threadIdx.y][threadIdx.x] = 0.0f;

//         // load tile N
//         int nRow = phase * TILE_WIDTH + threadIdx.y;
//         if (nRow < n && col < o)
//             Ntile[threadIdx.y][threadIdx.x] = N[nRow * o + col];
//         else
//             Ntile[threadIdx.y][threadIdx.x] = 0.0f;

//         __syncthreads(); // Wait until the whole tile is loaded

//         // Multiply tile rows and cols
//         for (int k = 0; k < TILE_WIDTH; k++)
//             value += Mtile[threadIdx.y][k] * Ntile[k][threadIdx.x];

//         __syncthreads(); // Wait before overwriting tiles
//     }

//     if (row < m && col < o)
//         P[row * o + col] = value;
// }


__global__ void TiledMatrixMulKernel(float* M, float* N, float* P,
                                     int m, int n, int o) {

    __shared__ float Mtile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ntile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float partialSum = 0.0f;

    // Number of tiles along the shared dimension
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < numTiles; phase++) {
        int tileIdx = phase * TILE_WIDTH;
        int localRow = threadIdx.y;
        int localCol = threadIdx.x;

        // Load M tile
        if (row < m && (tileIdx + localCol) < n) {
            Mtile[localRow][localCol] = M[row * n + (tileIdx + localCol)];
        } else {
            Mtile[localRow][localCol] = 0.0f;
        }

        // Load N tile 
        if ((tileIdx + localRow) < n && col < o) {
            Ntile[localRow][localCol] = N[(tileIdx + localRow) * o + col];
        } else {
            Ntile[localRow][localCol] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            partialSum += Mtile[localRow][k] * Ntile[k][localCol];
        }

        __syncthreads(); 
    }

    if (row < m && col < o) {
        P[row * o + col] = partialSum;
    }
}



void matrixMul(float* M, float* N, float* P, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, m * n * sizeof(float));
    cudaMalloc((void**)&d_N, n * o * sizeof(float));
    cudaMalloc((void**)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

void matrixMulTiling(float* M, float* N, float* P, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, m * n * sizeof(float));
    cudaMalloc((void**)&d_N, n * o * sizeof(float));
    cudaMalloc((void**)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

float benchmark(void (*func)(float*, float*, float*, int, int, int), float* M, float* N, float* P, int m, int n, int o,
                int warmup = 2, int reps = 10) {
    for (int i = 0; i < warmup; ++i) {
        func(M, N, P, m, n, o);
    }

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);

    float totalTime_ms = 0.0f;

    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(iterStart);
        func(M, N, P, m, n, o);
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);

        float iterTime = 0.0f;
        cudaEventElapsedTime(&iterTime, iterStart, iterStop);
        totalTime_ms += iterTime;

        clear_l2();
    }

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);

    return totalTime_ms / reps;
}

bool allclose(float* M, float* N, int m, int n, float tol = 1e-5) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabs(M[i * n + j] - N[i * n + j]) > tol) {
                return false;
            }
        }
    }
    return true;
}

void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(6) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int m = 10000, n = 10000, o = 10000;

    float* M = new float[m * n];
    float* N = new float[n * o];
    float* P1 = new float[m * o];
    float* P2 = new float[m * o];

    for (int i = 0; i < m * n; ++i) {
        M[i] = static_cast<float>(1);
    }
    for (int i = 0; i < n * o; ++i) {
        N[i] = static_cast<float>(1.5);
    }

    float avgTimeMatrixMulTiling = benchmark(matrixMulTiling, M, N, P1, m, n, o);
    std::cout << "Average time for matrixMulTiling: " << avgTimeMatrixMulTiling << " ms" << std::endl;

    float avgTimeMatrixMul = benchmark(matrixMul, M, N, P2, m, n, o);
    std::cout << "Average time for matrixMul: " << avgTimeMatrixMul << " ms" << std::endl;

    bool same = allclose(P1, P2, m, o);
    std::cout << "Output values are " << (same ? "the same" : "different") << std::endl;

    delete[] M;
    delete[] N;
    delete[] P1;
    delete[] P2;

    return 0;
}