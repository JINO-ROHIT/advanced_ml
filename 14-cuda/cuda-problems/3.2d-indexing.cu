// single block
// 2d indexing to element wise add.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int WIDTH = 3;   // 3 columns
int HEIGHT = 2;  // 2 rows

__global__ void add(float *A, float *B, float *C, int width, int height){
    int col = threadIdx.x;  // x-coordinate (column) 0-2
    int row = threadIdx.y;  // y-coordinate (row) 0-1
    
    int idx = row * width + col;
    
    if(col < width && row < height){
        C[idx] = A[idx] + B[idx];
        printf("Thread (%d,%d) -> idx %d: C[%d] = %.0f\n", 
               col, row, idx, idx, C[idx]);
    }
}

int main() {
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    int total_elements = WIDTH * HEIGHT;  // 6 elements total
    
    h_A = new float[total_elements];
    h_B = new float[total_elements];
    h_C = new float[total_elements];
    
    // think of it as 3x2 matrix)
    for (int i = 0; i < total_elements; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }
    
    cudaMalloc(&d_A, sizeof(float) * total_elements);
    cudaMalloc(&d_B, sizeof(float) * total_elements);
    cudaMalloc(&d_C, sizeof(float) * total_elements);
    
    cudaMemcpy(d_A, h_A, sizeof(float) * total_elements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * total_elements, cudaMemcpyHostToDevice);
    
    // Launch kernel with 2D thread block: 3 x 2 = 6 threads
    dim3 threadsPerBlock(WIDTH, HEIGHT);  // 3 x 2 threads
    dim3 numBlocks(1, 1);                 // 1 x 1 blocks
    
    add<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH, HEIGHT);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, sizeof(float) * total_elements, cudaMemcpyDeviceToHost);
    
    printf("\nResults - 3x2 matrix:\n");
    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            int idx = row * WIDTH + col;
            printf("%6.0f ", h_C[idx]);
        }
        printf("\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}