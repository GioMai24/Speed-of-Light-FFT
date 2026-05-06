#include "utilsCu.cuh"
#include <omp.h>
#include <cuda/cmath>
#include <cuda/std/complex>
#include <cuda/std/numbers>
#include <iostream>
#include <fstream>
#include <string>

/** @file
 * @brief Double GPU CUDA implementation.
 *
 * Compute "100 images" in parallel on two GPUs. Easily extendible to more GPUs by tinkering some parameters.
 */


int main(int argc, char **argv){
    if (argc != 2){
        std::cout << "Give me some dimensions!" << std::endl;
        return 1;
    }

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount != 2){
		std::cout << "The 2 GPUs were not detected, be afraid." << std::endl;
		return 2;
	}

	omp_set_num_threads(deviceCount);

	// FILES
    const bool saveData = false;
    std::ifstream load[deviceCount];
    std::ofstream save;

    cudaStream_t stream[deviceCount];
	for (int i=0; i<deviceCount; i++){
		cudaSetDevice(i);
	    cudaStreamCreate(&stream[i]);
	}

	// GRID (NOT THE CUDA ONE!)
	const std::string sRows = argv[1];
	const int rows = std::stoi(sRows);
	const int cols = rows;
	const size_t size = rows * cols;
	const size_t cuSize = size * sizeof(cuda::std::complex<float>);
    cuda::std::complex<float> *grid[deviceCount];
	cuda::std::complex<float> *Dgrid[deviceCount];
	cuda::std::complex<float> *DgridT[deviceCount];

	for (int i=0; i<deviceCount; i++){
		cudaSetDevice(i);
		cudaMallocAsync(&Dgrid[i], cuSize, stream[i]);
		cudaMallocAsync(&DgridT[i], cuSize, stream[i]);
	}

	// CUGRIDS (THE CUDA ONES!)
    const int threadsXBlock = cols <= 1024 ? (cols >> 1) : 1024;
    const int blockCols = (cols >> 1) / threadsXBlock;
    dim3 blocks(blockCols, rows);
    dim3 blocksR(blockCols*2, rows);
    dim3 threadsXBlockT(32, 32);
    const int bColsT = cols >> 5, bRowsT = rows >> 5;
    dim3 blocksT(bColsT, bRowsT);
    dim3 blocksC(bColsT >> 1, bRowsT);

	for (int i=0; i<deviceCount; i++){
		cudaSetDevice(i);
		cudaMallocHost(&grid[i], cuSize, cudaHostAllocDefault);
	}

	// INPUT FILES
	std::string names[2] = {"data/" + sRows + ".bin", "data/" + sRows + "Return.bin"};
	#pragma omp parallel for schedule(static, 1)
	for (int i=0; i<deviceCount; i++){
		cudaSetDevice(i);
		for (int counter=0; counter<50; counter++){
			load[i].open(names[i], std::ios::binary | std::ios::ate);
			std::streamsize nChar = load[i].tellg();
			load[i].seekg(0);
			load[i].read(reinterpret_cast<char *> (grid[i]), nChar);
			load[i].close();

			cudaMemcpyAsync(Dgrid[i], grid[i], cuSize, cudaMemcpyHostToDevice, stream[i]);

			centerKer<<<blocksC, threadsXBlockT, 0, stream[i]>>>(Dgrid[i], cols);

			// FFT ROWS
			revBitOrdKer<<<blocksR, threadsXBlock, 0, stream[i]>>>(Dgrid[i], DgridT[i], cols);
			for(int s=1; s<=log2(cols); s++){
				int m = 1 << s;
				coolSubKer<<<blocks, threadsXBlock, 0, stream[i]>>>(DgridT[i], m, cols);
			}

			// FFT COLS
			sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream[i]>>>(DgridT[i], Dgrid[i], cols);
			revBitOrdKer<<<blocksR, threadsXBlock, 0, stream[i]>>>(Dgrid[i], DgridT[i], cols);
			for(int s=1; s<=log2(cols); s++){
				int m = 1 << s;
				coolSubKer<<<blocks, threadsXBlock, 0, stream[i]>>>(DgridT[i], m, cols);
			}
			sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream[i]>>>(DgridT[i], Dgrid[i], cols);

			// GAUSSIAN BLUR
			gaussKer<<<blocksT, threadsXBlockT, 0, stream[i]>>>(Dgrid[i], cols, rows,  1.f / (2.f * 80.f * 80.f));

			// IFFT ROWS
			revBitOrdKer<<<blocksR, threadsXBlock, 0, stream[i]>>>(Dgrid[i], DgridT[i], cols);
			for(int s=1; s<=log2(cols); s++){
				int m = 1 << s;
				busLoocKer<<<blocks, threadsXBlock, 0, stream[i]>>>(DgridT[i], m, cols);
			}

			// IFFT COLS
			sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream[i]>>>(DgridT[i], Dgrid[i], cols);
			revBitOrdKer<<<blocksR, threadsXBlock, 0, stream[i]>>>(Dgrid[i], DgridT[i], cols);
			for(int s=1; s<=log2(cols); s++){
				int m = 1 << s;
				busLoocKer<<<blocks, threadsXBlock, 0, stream[i]>>>(DgridT[i], m, cols);
			}
			sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream[i]>>>(DgridT[i], Dgrid[i], cols);
			mulKer<<<blocksT, threadsXBlockT, 0, stream[i]>>>(Dgrid[i], cols, 1.f / (float) size);

			centerKer<<<blocksC, threadsXBlockT, 0, stream[i]>>>(Dgrid[i], cols);
			cudaMemcpyAsync(grid[i], Dgrid[i], cuSize, cudaMemcpyDeviceToHost, stream[i]);
			cudaStreamSynchronize(stream[i]);
		}
	}

    if (saveData){
		std::string outNames[2] = {"data/fftCu.bin", "data/fftCu2.bin"};
		for (int someName=0; someName<deviceCount; someName++){
            // REAL CASE, NOT FOR COMPLEX ORIGINAL IMAGE!!!!
			float *saveFft = new float[size];
			for(int i=0; i<size; i++){
//	            saveFft[i] = log(1.f + hypotf(grid[i].real(), grid[i].imag()));
//	            saveFft[i] = hypotf(grid[i].real(), grid[i].imag());
				saveFft[i] = grid[someName][i].real();
			}

			// COMPLEX CASE
//	        std::complex<float> *saveFft = new std::complex<float>[size];
//	        for(int i=0; i<size; i++){
//	                saveFft[i] = grid[i];
//	        }

			save.open(outNames[someName], std::ios::binary);
			save.write(reinterpret_cast<char *> (saveFft), size*sizeof(float) / sizeof(char));
	//        save.write(reinterpret_cast<char *> (saveFft), size*sizeof(std::complex<float>) / sizeof(char));
			save.close();
			delete[] saveFft;
		}
    }

	#pragma omp parallel for schedule(static, 1)
	for (int i=0; i<deviceCount; i++){
		cudaSetDevice(i);
		cudaFreeAsync(Dgrid[i], stream[i]);
		cudaFreeAsync(DgridT[i], stream[i]);
		cudaFreeAsync(grid[i], stream[i]);
		cudaStreamSynchronize(stream[i]);
		cudaStreamDestroy(stream[i]);
	}

    return 0;
}
