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
    bool saveData = true;
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
	const int rows = 1024;
	const int cols = 1024;
	const int size = rows * cols;
	const int floatSize = size * sizeof(cufftReal);
	const int cuSize = rows * (cols/2 + 1) * sizeof(cufftComplex);
    cufftReal *grid = nullptr;
	cufftReal *Dgrid = nullptr;
	cufftComplex *Dres = nullptr;
	cufftComplex *res = nullptr;
	cudaMallocHost(&grid, floatSize, cudaHostAllocDefault);
	cudaMalloc(&Dgrid, floatSize);
	cudaMalloc(&Dres, cuSize);
	cudaMallocHost(&res, cuSize, cudaHostAllocDefault);
    cufftPlan2d(&plan, rows, cols, CUFFT_R2C);
    cufftSetStream(plan, stream);

	load.open("data/data.bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (grid), nChar);
	load.close();

//    if (saveData){
//        centerSpectrum(grid, rows, cols);
//    }
	cudaMemcpyAsync(Dgrid, grid, floatSize, cudaMemcpyHostToDevice, stream);

	cudaEventRecord(cuT1, stream);
	cufftExecR2C(plan, Dgrid, Dres);
	cudaEventRecord(cuT2, stream);
	cudaMemcpyAsync(res, Dres, cuSize, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

    // spectrum then log scale
    if (saveData){
        float *saveFft = new float[rows * (cols/2 + 1)];
        for(int i=0; i<(rows * cols/2 + 1); i++){
            saveFft[i] = log(1.f + hypotf(res[i].x, res[i].y));
        }

        save.open("data/fftCu.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (saveFft), rows*(cols/2 + 1)*sizeof(float) / sizeof(char));
        save.close();
        delete[] saveFft;
    }


	cudaEventElapsedTime(&cuDt, cuT1, cuT2);
	std::cout << "Time: " << cuDt << std::endl;

	cudaFreeHost(grid);
	cudaFreeHost(res);
	cudaFree(Dgrid);
	cudaFree(Dres);
    cudaEventDestroy(cuT1);
    cudaEventDestroy(cuT2);
    cufftDestroy(plan);
	cudaStreamDestroy(stream);

    return 0;
}
