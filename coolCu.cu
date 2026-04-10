#include <iostream>
#include <fstream>
#include <string>
#include <cuda/cmath>
#include <cuda/std/complex>
#include <cuda/std/numbers>
#include <chrono>
#include "utils.h"

__global__ void revBitOrdKer(cuda::std::complex<float> *in, cuda::std::complex<float> *out, const int cols){
    int xId = blockDim.x * blockIdx.x + threadIdx.x;
    int n = xId;
    const int lCols = cuda::ilog2(cols);
    int rXId = 0;
    for(int i=0; i<lCols; i++){
        rXId <<= 1;
        rXId |= (n & 1);
        n >>= 1;
    }
    out[blockIdx.y * cols + rXId] = in[blockIdx.y * cols + xId];
}

/**
 * <<<rows, cols>>> limited by max threads x block, and shared memory definition...
 */
__global__ void revBitShOrdKer(cuda::std::complex<float> *in, cuda::std::complex<float> *out, const int cols){
    __shared__ cuda::std::complex<float> help[1024];  // Oh to maually change this... (cols)
    help[threadIdx.x] = in[blockIdx.x * cols + threadIdx.x];
    int xId = threadIdx.x;
    int n = xId;
    const int lCols = cuda::ilog2(cols);
    int rXId = 0;
    for(int i=0; i<lCols; i++){
        rXId <<= 1;
        rXId |= (n & 1);
        n >>= 1;
    }
    __syncthreads();
    out[blockIdx.x * cols + xId] = help[rXId];
}


/**
 * Set <<<(grid.x, rows), threads>>> where grid.x * threads = cols/2
 */
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


template<typename T>
__global__ void sharedTransposeKer(T *in, T *out, const int cols){
    __shared__ T helper[32][33];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    helper[threadIdx.x][threadIdx.y] = in[row * cols + col];
    __syncthreads();
    out[(blockIdx.x * blockDim.x + threadIdx.y) * cols + (blockIdx.y * blockDim.y + threadIdx.x)] = helper[threadIdx.y][threadIdx.x];
}

template<typename T>
__global__ void transposeKer(T *in, T *out, const int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    out[row * cols + col] = in[col * cols + row];
}



int main(int argc, char **argv){
    using namespace std::chrono;

	// files
    bool saveData = false;
    std::ifstream load;
    std::ofstream save;

    // cuda
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t cuT1, cuT2;
    cudaEventCreate(&cuT1);
    cudaEventCreate(&cuT2);
    float cuDt;


	// grid
	const int rows = 16384;
	const int cols = rows;
	const size_t size = rows * cols;
	const size_t cuSize = size * sizeof(cuda::std::complex<float>);
    cuda::std::complex<float> *grid = nullptr;
	cuda::std::complex<float> *Dgrid = nullptr;
	cuda::std::complex<float> *DgridT = nullptr;
	cudaMallocHost(&grid, cuSize, cudaHostAllocDefault);
	cudaMalloc(&Dgrid, cuSize);
	cudaMalloc(&DgridT, cuSize);


	load.open("data/16384.bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (grid), nChar);
	load.close();

    if (saveData){
        centerSpectrum(grid, rows, cols);
    }
	cudaMemcpyAsync(Dgrid, grid, cuSize, cudaMemcpyHostToDevice, stream);

    // fft rows
//    const int blockCols = 16;
//    const int threadsXBlock = cols / 2 / blockCols;
    const int threadsXBlock = cols <= 1024 ? cols / 2 : 1024;
    const int blockCols = cols / 2 / threadsXBlock;
    dim3 blocks(blockCols, rows);
    dim3 blocksR(blockCols*2, rows);
    // only works for N >= 1024
    dim3 blocksT(cols / 32, rows / 32);
    dim3 threadsXBlockT(32, 32);

	cudaEventRecord(cuT1, stream);
    revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
    for(int s=1; s<=log2(cols); s++){
        int m = 1 << s;
        coolSubKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
    }

    // fft cols (works because square matrix...)
    sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
//    transposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
//    revBitShOrdKer<<<1024, 1024, 0, stream>>>(Dgrid, DgridT, cols);
    revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
    for(int s=1; s<=log2(cols); s++){
        int m = 1 << s;
        coolSubKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
    }
    sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
//    transposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
    cudaEventRecord(cuT2, stream);
    cudaMemcpyAsync(grid, Dgrid, cuSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // spectrum then log scale
    if (saveData){
        float *saveFft = new float[size];
        for(int i=0; i<size; i++){
            saveFft[i] = log(1.f + hypotf(grid[i].real(), grid[i].imag()));
        }

        save.open("data/fftCu.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (saveFft), size*sizeof(float) / sizeof(char));
        save.close();
        delete[] saveFft;
    }


	cudaEventElapsedTime(&cuDt, cuT1, cuT2);
	std::cout << "Time: " << cuDt << std::endl;

	cudaFreeHost(grid);
	cudaFree(Dgrid);
	cudaFree(DgridT);
    cudaEventDestroy(cuT1);
    cudaEventDestroy(cuT2);
	cudaStreamDestroy(stream);

    return 0;
}
