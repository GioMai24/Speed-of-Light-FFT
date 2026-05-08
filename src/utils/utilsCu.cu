#include "utilsCu.cuh"
#include <cuda/std/complex>
#include <cuda/cmath>


__global__ void revBitOrdKer(cuda::std::complex<float> *in, cuda::std::complex<float> *out, const int cols){
    int xId = blockDim.x * blockIdx.x + threadIdx.x;
    int n = xId;
    int rXId = 0;
    for(int i=0; i<cuda::ilog2(cols); i++){
        rXId <<= 1;
        rXId |= (n & 1);
        n >>= 1;
    }
    out[blockIdx.y * cols + rXId] = in[blockIdx.y * cols + xId];
}

__global__ void revBitShOrdKer(cuda::std::complex<float> *in, cuda::std::complex<float> *out, const int cols, const int iters){
    extern __shared__ cuda::std::complex<float> help[];
    for (int i=0; i<iters; i++){
        help[threadIdx.x * iters + i] = in[blockIdx.y * cols + threadIdx.x * iters + i];
    }
    int xId = blockDim.x * blockIdx.x + threadIdx.x;
    int n = xId;
    int rXId = 0;
    for(int i=0; i<cuda::ilog2(cols); i++){
        rXId <<= 1;
        rXId |= (n & 1);
        n >>= 1;
    }
    __syncthreads();
    out[blockIdx.y * cols + xId] = help[rXId];
}


__global__ void coolSubKer(cuda::std::complex<float> *res, const int m, const int cols){
    int xId = blockDim.x * blockIdx.x + threadIdx.x;
    int j = xId & ((m >> 1) - 1);  // modulo
    int k = (2 * xId >> cuda::ilog2(m)) * m;
    cuda::std::complex<float> w = cuda::std::polar(1.f, -2 * j * cuda::std::numbers::pi_v<float> / (float) m);

    cuda::std::complex<float> t = w * res[k + j + (m >> 1) + blockIdx.y * cols];
    cuda::std::complex<float> u = res[k + j + blockIdx.y * cols];
    res[k + j + blockIdx.y * cols] = u + t;
    res[k + j + (m >> 1) + blockIdx.y * cols] = u - t;
}

__global__ void busLoocKer(cuda::std::complex<float> *res, const int m, const int cols){
    int xId = blockDim.x * blockIdx.x + threadIdx.x;
    int j = xId & ((m >> 1) - 1);  // modulo
    int k = (2 * xId >> cuda::ilog2(m)) * m;
    cuda::std::complex<float> w = cuda::std::polar(1.f, 2 * j * cuda::std::numbers::pi_v<float> / (float) m);

    cuda::std::complex<float> t = w * res[k + j + (m >> 1) + blockIdx.y * cols];
    cuda::std::complex<float> u = res[k + j + blockIdx.y * cols];
    res[k + j + blockIdx.y * cols] = u + t;
    res[k + j + (m >> 1) + blockIdx.y * cols] = u - t;
}
