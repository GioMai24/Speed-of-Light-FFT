#ifndef UTILS_CUH1758
#define UTILS_CUH1758
#include <cuda/std/complex>

/** @file
 * @brief CUDA kernels.
 *
 * Kernels used in the DFT CUDA implementation. They should appear in this file in roughly the same order they appear in the main code.
 */



/**
 * @brief Center image spectrum.
 *
 * Multiplies by -1 every other "bidimensional" array element in a chessboard fashion.
 * Threads needed: (half cols, rows).
 *
 * @tparam T array datatype.
 * @param[in,out] arr array.
 * @param[in] cols columns of the array.
 */
template<typename T>
__global__ void centerKer(T *arr, const int cols){
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    const int xId = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + (yId & 1);
    arr[yId * cols + xId] *= -1;
}


/**
 * @brief Perform rowwise Reverse Bit Ordering of an array.
 *
 * @param[in] in input array.
 * @param[out] out output array.
 * @param[in] cols columns of the array.
 */
__global__ void revBitOrdKer(cuda::std::complex<float> *in, cuda::std::complex<float> *out, const int cols);

/**
 * @brief Perform rowwise Reverse Bit Ordering of an array with shared memory (deprecated).
 *
 * A single block would need a whole row in the shared memory. This both requires too much memory and multiple accesses from a single thread if cols>1024.
 *
 * @param[in] in input array.
 * @param[out] out output array.
 * @param[in] cols columns of the array.
 */
__global__ void revBitShOrdKer(cuda::std::complex<float> *in, cuda::std::complex<float> *out, const int cols);


/**
 * @brief DFT substep.
 *
 * Given m from the stride, each thread takes care of a k-j couple in the bidimensional array.
 * Threads needed: <<<(grid.x, rows), threads>>> where grid.x * threads = cols/2.
 *
 * @param[in,out] res array.
 * @param[in] m 2^{stride} parameter of the butterfly operation.
 * @param[in] cols columns of the array.
 */
__global__ void coolSubKer(cuda::std::complex<float> *res, const int m, const int cols);

/**
 * @brief IDFT substep. Like DFT but the inverse.
 *
 * Given m from the stride, each thread takes care of a k-j couple in the bidimensional array.
 * Threads needed: <<<(grid.x, rows), threads>>> where grid.x * threads = cols/2.
 *
 * @param[in,out] res array.
 * @param[in] m 2^{stride} parameter of the butterfly operation.
 * @param[in] cols columns of the array.
 */
__global__ void busLoocKer(cuda::std::complex<float> *res, const int m, const int cols);



 /**
 * @brief Gaussian Filtering.
 *
 * The kernel assumes the input array's spectrum has been centered.
 *
 * @tparam T array datatype.
 * @param[in,out] res input array.
 * @param[in] cols colums of the array.
 * @param[in] rows rows of the array.
 * @param[in] rS precomputed reciprocal of the filter's sigma factor.
 */
template<typename T>
__global__ void gaussKer(T *res, const int cols, const int rows, const float rS){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = xId - (cols >> 1);
    const int y = yId - (rows>> 1);
    res[yId * cols + xId] *= cuda::std::exp(- (float)(x*x + y*y) * rS);
}


/**
 * @brief Multiplication kernel.
 *
 * @tparam T array datatype.
 * @param[in,out] res input array.
 * @param[in] cols colums of the array.
 * @param[in] rN factor to multiply for.
 */
template<typename T>
__global__ void mulKer(T *res, const int cols, const float rN){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    res[yId * cols + xId] *= rN;
}


/**
 * @brief 2 arrays subtraction kernel.
 *
 * (Not used).
 *
 * @tparam T arrays datatype.
 * @param[in,out] arr1 input array.
 * @param[in] arr2 array to subtract.
 * @param[in] cols colums of the array.
 */
template<typename T>
__global__ void subKer(T *arr1, T *arr2, const int cols){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    arr1[yId * cols + xId] -= arr2[yId*cols + xId];
}


/**
 * @brief Transposition kernel.
 *
 * @tparam T arrays datatype.
 * @param[in] in input array.
 * @param[out] out transposed array.
 * @param[in] cols columns of the array.
 */
template<typename T>
__global__ void transposeKer(T *in, T *out, const int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    out[row * cols + col] = in[col * cols + row];
}

/**
 * @brief Transposition kernel with shared memory.
 *
 * @tparam T arrays datatype.
 * @param[in] in input array.
 * @param[out] out transposed array.
 * @param[in] cols columns of the array.
 */
template<typename T>
__global__ void sharedTransposeKer(T *in, T *out, const int cols){
    __shared__ T helper[32][33];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    helper[threadIdx.x][threadIdx.y] = in[row * cols + col];
    __syncthreads();
    out[(blockIdx.x * blockDim.x + threadIdx.y) * cols + (blockIdx.y * blockDim.y + threadIdx.x)] = helper[threadIdx.y][threadIdx.x];
}

#endif
