#include <iostream>
#include <fstream>
#include <string>
#include <cuda/cmath>
#include <cuda/std/complex>
#include <cuda/std/numbers>
#include <chrono>
#include <cufft.h>
#include "utils.h"


int main(int argc, char **argv){

	// files
    bool saveData = false;
    std::ifstream load;
    std::ofstream save;

    // cuda
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cufftHandle plan;

    cudaEvent_t cuT1, cuT2;
    cudaEventCreate(&cuT1);
    cudaEventCreate(&cuT2);
    float cuDt;


	// grid
	const int rows = 8192;
	const int cols = rows;
	const int size = rows * cols;
	const int cuSize = size  * sizeof(cufftComplex);
    cufftComplex *grid = nullptr;
	cufftComplex *Dgrid = nullptr;
	cudaMallocHost(&grid, cuSize, cudaHostAllocDefault);
	cudaMalloc(&Dgrid, cuSize);
    cufftPlan2d(&plan, rows, cols, CUFFT_C2C);
    cufftSetStream(plan, stream);

	load.open("data/8192.bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (grid), nChar);
	load.close();

//    if (saveData){
//        centerSpectrum(grid, rows, cols);
//    }
	cudaMemcpyAsync(Dgrid, grid, cuSize, cudaMemcpyHostToDevice, stream);

	cudaEventRecord(cuT1, stream);
	cufftExecC2C(plan, Dgrid, Dgrid, CUFFT_FORWARD);
	cudaEventRecord(cuT2, stream);
	cudaMemcpyAsync(grid, Dgrid, cuSize, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

    // spectrum then log scale
    if (saveData){
        float *saveFft = new float[size];
        for(int i=0; i<size; i++){
            saveFft[i] = log(1.f + hypotf(grid[i].x, grid[i].y));
        }

        save.open("data/iFftCufft.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (saveFft), size * sizeof(float) / sizeof(char));
        save.close();
        delete[] saveFft;
    }


	cudaEventElapsedTime(&cuDt, cuT1, cuT2);
	std::cout << "Time: " << cuDt << std::endl;

	cudaFreeHost(grid);
	cudaFree(Dgrid);
    cudaEventDestroy(cuT1);
    cudaEventDestroy(cuT2);
    cufftDestroy(plan);
	cudaStreamDestroy(stream);

    return 0;
}
