/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
 #define BLOCK_SIZE 512


__global__ void naiveReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/
    __shared__ float partialSum[BLOCK_SIZE*4];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    if(t<=size){
        partialSum[t] = in[start + t];
        partialSum[blockDim.x+t] = in[start + blockDim.x+t];
    }else{
        partialSum[t]=0;
    }
     __syncthreads();
     for (unsigned int stride = 1;stride <= blockDim.x; stride *= 2)
        {
        __syncthreads();
        if (t % stride == 0) partialSum[2*t]+= partialSum[2*t+stride];
        }
     __syncthreads();
   
     if (t == 0) out[blockIdx.x] = partialSum[0];
}

__device__ void warpReduce(volatile float* sdata, int tid) {
sdata[tid] += sdata[tid + 32];
sdata[tid] += sdata[tid + 16];
sdata[tid] += sdata[tid + 8];
sdata[tid] += sdata[tid + 4];
sdata[tid] += sdata[tid + 2];
sdata[tid] += sdata[tid + 1];
}


__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // OPTIMIZED REDUCTION IMPLEMENTATION
    __shared__ float partialSum[BLOCK_SIZE*4];
    unsigned int t = threadIdx.x;
    unsigned int start = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    partialSum[t] = in[start] + in[start+blockDim.x];

     __syncthreads();
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
       if (t < s) partialSum[t] += partialSum[t + s];
       __syncthreads();
    }
    if (t < 32) warpReduce(partialSum, t);
   
    if (t == 0) out[blockIdx.x] = partialSum[0];
}
