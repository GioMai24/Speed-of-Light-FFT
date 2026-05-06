#include <cufft.h>
#include <cuda/std/cmath>
#include <iostream>
#include <fstream>
#include <string>

/** @file
 * @brief CUFFT library implementation.
 *
 * Compute 100 "images". Change counter var in image loop according to number of streams...
 * Differently from the standard CUDA implementation, all the needed functions are defined here. This is due to the usage of cufftComplex specific type.
 */



/**
 * @brief Center image spectrum.
 *
 * Multiplies by -1 every other "bidimensional" array element in a chessboard fashion.
 * Threads needed: (half cols, rows).
 *
 * @param arr input array.
 * @param cols columns of the array.
 */
__global__ void centerKer(cufftComplex *arr, const int cols){
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    const int xId = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + (yId & 1);

    arr[yId * cols + xId].x *= -1;
    arr[yId * cols + xId].y *= -1;
}

 /**
 * @brief Gaussian Filtering.
 *
 * The kernel assumes the input array's spectrum has been centered.
 *
 * @param res input array.
 * @param cols colums of the array.
 * @param rows rows of the array.
 * @param rS precomputed reciprocal of the filter's sigma factor.
 */
__global__ void gaussKer(cufftComplex *res, const int cols, const int rows, const float rS){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = xId - (cols >> 1);
    const int y = yId - (rows >> 1);

    const float fac = cuda::std::exp(- (float)(x*x + y*y) * rS);
    res[yId * cols + xId].x *= fac;
    res[yId * cols + xId].y *= fac;
}

/**
 * @brief Multiplication kernel.
 *
 * @param res input array.
 * @param cols colums of the array.
 * @param rN factor to multiply for.
 */
__global__ void mulKer(cufftComplex *res, const int cols, const float rN){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;

    res[yId * cols + xId].x *= rN;
    res[yId * cols + xId].y *= rN;
}


int main(int argc, char **argv){
    if (argc != 2){
        std::cout << "Give me some dimensions!" << std::endl;
        return 1;
    }

	// FILES
    const bool saveData = false;
    std::ifstream load;
    std::ofstream save;

    cudaStream_t stream, stream2;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&stream2);
    cufftHandle plan, plan2;

	// GRID (NOT THE CUDA ONE!)
	const std::string sRows = argv[1];
	const int rows = std::stoi(sRows);
	const int cols = rows;
	const size_t size = rows * cols;
	const size_t cuSize = size  * sizeof(cufftComplex);
    cufftComplex *grid = nullptr;
    cufftComplex *grid2 = nullptr;
	cufftComplex *Dgrid = nullptr;
	cufftComplex *Dgrid2 = nullptr;
	cudaMallocAsync(&Dgrid, cuSize, stream);
	cudaMallocAsync(&Dgrid2, cuSize, stream);

    cufftPlan2d(&plan, rows, cols, CUFFT_C2C);
    cufftPlan2d(&plan2, rows, cols, CUFFT_C2C);
    cufftSetStream(plan, stream);
    cufftSetStream(plan2, stream2);

	// CUGRIDS (THE CUDA ONES!)
	const int bColsC = cols >> 6, bRows = rows >> 5;
    dim3 threadsXBlock(32, 32);
    dim3 blocksC(bColsC, bRows);
    dim3 blocksG(bColsC << 1, bRows);

	cudaMallocHost(&grid, cuSize, cudaHostAllocDefault);
	cudaMallocHost(&grid2, cuSize, cudaHostAllocDefault);
	for (int counter=0; counter<50; counter++)
    {
        load.open("data/" + sRows + ".bin", std::ios::binary | std::ios::ate);
        std::streamsize nChar = load.tellg();
        load.seekg(0);
        load.read(reinterpret_cast<char *> (grid), nChar);
        load.close();

        load.open("data/" + sRows + "Return.bin", std::ios::binary | std::ios::ate);
        nChar = load.tellg();
        load.seekg(0);
        load.read(reinterpret_cast<char *> (grid2), nChar);
        load.close();

        cudaMemcpyAsync(Dgrid, grid, cuSize, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Dgrid2, grid2, cuSize, cudaMemcpyHostToDevice, stream2);

        centerKer<<<blocksC, threadsXBlock, 0, stream>>>(Dgrid, cols);
        centerKer<<<blocksC, threadsXBlock, 0, stream2>>>(Dgrid2, cols);

        cufftExecC2C(plan, Dgrid, Dgrid, CUFFT_FORWARD);
        cufftExecC2C(plan2, Dgrid2, Dgrid2, CUFFT_FORWARD);

        gaussKer<<<blocksG, threadsXBlock, 0, stream>>>(Dgrid, cols, rows,  1.f / (2.f * 80.f * 80.f));
        gaussKer<<<blocksG, threadsXBlock, 0, stream2>>>(Dgrid2, cols, rows,  1.f / (2.f * 80.f * 80.f));

        cufftExecC2C(plan, Dgrid, Dgrid, CUFFT_INVERSE);
        cufftExecC2C(plan2, Dgrid2, Dgrid2, CUFFT_INVERSE);
        mulKer<<<blocksG, threadsXBlock, 0, stream>>>(Dgrid, cols, 1.f / (float) size);
        mulKer<<<blocksG, threadsXBlock, 0, stream2>>>(Dgrid2, cols, 1.f / (float) size);

        centerKer<<<blocksC, threadsXBlock, 0, stream>>>(Dgrid, cols);
        centerKer<<<blocksC, threadsXBlock, 0, stream2>>>(Dgrid2, cols);

        cudaMemcpyAsync(grid, Dgrid, cuSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(grid2, Dgrid2, cuSize, cudaMemcpyDeviceToHost, stream2);
        cudaStreamSynchronize(stream);
        cudaStreamSynchronize(stream2);
    }
    if (saveData){
        float *saveFft = new float[size];
        for(int i=0; i<size; i++){
//            saveFft[i] = log(1.f + hypotf(grid[i].x, grid[i].y));
            saveFft[i] = grid[i].x;
        }

        save.open("data/fftCufft.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (saveFft), size * sizeof(float) / sizeof(char));
        save.close();
        delete[] saveFft;
    }

	cudaFreeAsync(grid, stream);
	cudaFreeAsync(grid2, stream2);
	cudaFreeAsync(Dgrid, stream);
	cudaFreeAsync(Dgrid2, stream2);
    cufftDestroy(plan);
    cufftDestroy(plan2);
    cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
    cudaStreamSynchronize(stream2);
	cudaStreamDestroy(stream2);

    return 0;
}
