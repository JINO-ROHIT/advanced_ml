#include <stdio.h>

__global__ void hello(){
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main(){
    hello<<<3, 3>>>();   
    cudaDeviceSynchronize();
}



// launch with 3 block having 3 threads each
// wait for all threads to finish execution