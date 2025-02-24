#include <iostream>

#define M 3  // Rows in A
#define K 4  // Columns in A (Rows in B)
#define N 3  // Columns in B

__global__ void matrixMul(float *A, float *B, float *C, int m, int k, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void printMatrix(float *mat, int n, const char *name) {
    std::cout << name << ":\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << mat[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Host matrices with fixed values
    float h_A[M * K] = { 1,  2,  3,  4,  
        5,  6,  7,  8,  
        9, 10, 11, 12 };

    float h_B[K * N] = { 1, 2, 3,  
            4, 5, 6,  
            7, 8, 9,  
            10,11,12 };

    float h_C[M * N] = {0};

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Launch kernel with one block
    dim3 blockSize(M, N);  // Single block, full computation within
    matrixMul<<<1, blockSize>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print matrices for verification
    printMatrix(h_A, N, "Matrix A");
    printMatrix(h_B, N, "Matrix B");
    printMatrix(h_C, N, "Matrix C (Result)");

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
