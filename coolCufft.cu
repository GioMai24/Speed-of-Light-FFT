#include <iostream>
#include <fstream>
#include <string>
#include <cuda/std/cmath>
#include <cufft.h>

/**
 * Threads: (half cols, rows) needed.
 */
__global__ void centerKer(cufftComplex *arr, const int cols){
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    const int xId = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + (yId & 1);

    arr[yId * cols + xId].x *= -1;
    arr[yId * cols + xId].y *= -1;
}


__global__ void gaussKer(cufftComplex *res, const int cols, const int rows, const float rS){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = xId - (cols >> 1);
    const int y = yId - (rows >> 1);

    const float fac = cuda::std::exp(- (float)(x*x + y*y) * rS);
    res[yId * cols + xId].x *= fac;
    res[yId * cols + xId].y *= fac;
}

__global__ void mulKer(cufftComplex *res, const int cols, const float rN){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;

    res[yId * cols + xId].x *= rN;
    res[yId * cols + xId].y *= rN;
}


int main(int argc, char **argv){
    if (argc != 2){
        std::cout << "Something's off dude..." << std::endl;
        return 1;
    }
	// files
    bool saveData = false;
    std::ifstream load;
    std::ofstream save;

    // cuda
    cudaStream_t stream, stream2;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&stream2);
    cufftHandle plan, plan2;

//    cudaEvent_t cuT1, cuT2;
//    cudaEventCreate(&cuT1);
//    cudaEventCreate(&cuT2);
//    float cuDt;


	// grid
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

    //	cudaEventRecord(cuT1, stream);
        cufftExecC2C(plan, Dgrid, Dgrid, CUFFT_FORWARD);
        cufftExecC2C(plan2, Dgrid2, Dgrid2, CUFFT_FORWARD);
    //	cudaEventRecord(cuT2, stream);

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
    // spectrum then log scale
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


//	cudaEventElapsedTime(&cuDt, cuT1, cuT2);
//	std::cout << "Time: " << cuDt << std::endl;

	cudaFreeAsync(grid, stream);
	cudaFreeAsync(grid2, stream2);
	cudaFreeAsync(Dgrid, stream);
	cudaFreeAsync(Dgrid2, stream2);
//    cudaEventDestroy(cuT1);
//    cudaEventDestroy(cuT2);
    cufftDestroy(plan);
    cufftDestroy(plan2);
    cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
    cudaStreamSynchronize(stream2);
	cudaStreamDestroy(stream2);

    return 0;
}
