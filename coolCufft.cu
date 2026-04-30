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
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cufftHandle plan;

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
	cufftComplex *Dgrid = nullptr;
	cudaMallocAsync(&Dgrid, cuSize, stream);
    cufftPlan2d(&plan, rows, cols, CUFFT_C2C);
    cufftSetStream(plan, stream);
	const int bColsC = cols >> 6, bRows = rows >> 5;
    dim3 threadsXBlock(32, 32);
    dim3 blocksC(bColsC, bRows);
    dim3 blocksG(bColsC << 1, bRows);

	cudaMallocHost(&grid, cuSize, cudaHostAllocDefault);
    load.open("data/" + sRows + ".bin", std::ios::binary | std::ios::ate);
//    load.open("data/cats/cut4K2048.bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (grid), nChar);
	load.close();


	cudaMemcpyAsync(Dgrid, grid, cuSize, cudaMemcpyHostToDevice, stream);

    centerKer<<<blocksC, threadsXBlock, 0, stream>>>(Dgrid, cols);

//	cudaEventRecord(cuT1, stream);
	cufftExecC2C(plan, Dgrid, Dgrid, CUFFT_FORWARD);
//	cudaEventRecord(cuT2, stream);

    gaussKer<<<blocksG, threadsXBlock, 0, stream>>>(Dgrid, cols, rows,  1.f / (2.f * 80.f * 80.f));

    cufftExecC2C(plan, Dgrid, Dgrid, CUFFT_INVERSE);
    mulKer<<<blocksG, threadsXBlock, 0, stream>>>(Dgrid, cols, 1.f / (float) size);

    centerKer<<<blocksC, threadsXBlock, 0, stream>>>(Dgrid, cols);

	cudaMemcpyAsync(grid, Dgrid, cuSize, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

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
	cudaFreeAsync(Dgrid, stream);
//    cudaEventDestroy(cuT1);
//    cudaEventDestroy(cuT2);
    cufftDestroy(plan);
    cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

    return 0;
}
