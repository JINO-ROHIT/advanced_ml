#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define OUT_TILE_DIM_SMALL 8
#define IN_TILE_DIM_SMALL (OUT_TILE_DIM_SMALL + 2)

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// Stencil implementations
void stencil_3d_sequential(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4, int c5,
                           int c6);

void stencil_3d_parallel_shared_memory(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4,
            int c5, int c6);