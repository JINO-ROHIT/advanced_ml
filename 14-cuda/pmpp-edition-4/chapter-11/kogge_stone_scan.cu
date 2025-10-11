__global__ void scan(float* X, float* Y, unsigned int N){
    extern __shared__ float temp[]; // use for unknown size at compile
    unsigned int thid = threadIdx.x;

    if(thid < N){
        temp[thid] = X[thid]; //move to shared memory
    }
    else{
        temp[id] = 0.0;
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        float temp;
        __syncthreads();

        if(thid >= stride){
            temp = temp[tid] + temp[tid - stride];
        }

        __syncthreads();

        if(thid >= stride){
            temp[thid] = temp;
        }
    }

    if(thid < N){
        Y[thid] = temp[thid];
    }
}