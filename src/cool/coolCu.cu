#include "utilsCu.cuh"
#include <cuda/cmath>
#include <cuda/std/complex>
#include <cuda/std/numbers>
#include <iostream>
#include <fstream>
#include <string>

/** @file
 * @brief Bare CUDA DFT implementation.
 *
 * Compute x "images". Change counter var in image loop according to number of streams...
 */



int main(int argc, char **argv){
    if (argc != 2){
        std::cout << "Give me some dimensions!" << std::endl;
        return 1;
    }

	// FILES
    const bool saveData = false;
    std::ifstream load;
    std::ofstream save;

    cudaStream_t stream;
//    cudaStream_t stream2;
    cudaStreamCreate(&stream);
//    cudaStreamCreate(&stream2);

	// GRID (NOT THE CUDA ONE!)
	const std::string sRows = argv[1];
	const int rows = std::stoi(sRows);
	const int cols = rows;
	const size_t size = rows * cols;
	const size_t cuSize = size * sizeof(cuda::std::complex<float>);
    cuda::std::complex<float> *grid = nullptr;
//    cuda::std::complex<float> *grid2 = nullptr;
	cuda::std::complex<float> *Dgrid = nullptr;
//	cuda::std::complex<float> *Dgrid2 = nullptr;
	cuda::std::complex<float> *DgridT = nullptr;
//	cuda::std::complex<float> *DgridT2 = nullptr;

	cudaMallocAsync(&Dgrid, cuSize, stream);
//	cudaMallocAsync(&Dgrid2, cuSize, stream2);
	cudaMallocAsync(&DgridT, cuSize, stream);
//	cudaMallocAsync(&DgridT2, cuSize, stream2);

	// CUGRIDS (THE CUDA ONES!)
    const int threadsXBlock = cols <= 1024 ? (cols >> 1) : 1024;
    const int blockCols = (cols >> 1) / threadsXBlock;
    dim3 blocks(blockCols, rows);
    dim3 blocksR(blockCols*2, rows);
    dim3 threadsXBlockT(32, 32);
    const int bColsT = cols >> 5, bRowsT = rows >> 5;
    dim3 blocksT(bColsT, bRowsT);
    dim3 blocksC(bColsT >> 1, bRowsT);

	cudaMallocHost(&grid, cuSize, cudaHostAllocDefault);
//	cudaMallocHost(&grid2, cuSize, cudaHostAllocDefault);
    for (int counter=0; counter<1; counter++)
    {
        load.open("data/" + sRows + ".bin", std::ios::binary | std::ios::ate);
        std::streamsize nChar = load.tellg();
        load.seekg(0);
        load.read(reinterpret_cast<char *> (grid), nChar);
        load.close();

//        load.open("data/" + sRows + "Return.bin", std::ios::binary | std::ios::ate);
//        nChar = load.tellg();
//        load.seekg(0);
//        load.read(reinterpret_cast<char *> (grid2), nChar);
//        load.close();

        cudaMemcpyAsync(Dgrid, grid, cuSize, cudaMemcpyHostToDevice, stream);
//        cudaMemcpyAsync(Dgrid2, grid2, cuSize, cudaMemcpyHostToDevice, stream2);

        centerKer<<<blocksC, threadsXBlockT, 0, stream>>>(Dgrid, cols);
//        centerKer<<<blocksC, threadsXBlockT, 0, stream2>>>(Dgrid2, cols);

        // FFT ROWS
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
//        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream2>>>(Dgrid2, DgridT2, cols);
        for(int s=1; s<=log2(cols); s++){
            int m = 1 << s;
            coolSubKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
//            coolSubKer<<<blocks, threadsXBlock, 0, stream2>>>(DgridT2, m, cols);
        }

        // FFT COLS
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
//        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream2>>>(DgridT2, Dgrid2, cols);
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
//        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream2>>>(Dgrid2, DgridT2, cols);
        for(int s=1; s<=log2(cols); s++){
            int m = 1 << s;
            coolSubKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
//            coolSubKer<<<blocks, threadsXBlock, 0, stream2>>>(DgridT2, m, cols);
        }
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
//        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream2>>>(DgridT2, Dgrid2, cols);

        // GAUSSIAN BLUR
        gaussKer<<<blocksT, threadsXBlockT, 0, stream>>>(Dgrid, cols, rows,  1.f / (2.f * 80.f * 80.f));
//        gaussKer<<<blocksT, threadsXBlockT, 0, stream2>>>(Dgrid2, cols, rows,  1.f / (2.f * 80.f * 80.f));

        // IFFT ROWS
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
//        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream2>>>(Dgrid2, DgridT2, cols);
        for(int s=1; s<=log2(cols); s++){
            int m = 1 << s;
            busLoocKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
//            busLoocKer<<<blocks, threadsXBlock, 0, stream2>>>(DgridT2, m, cols);
        }

        // IFFT COLS
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
//        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream2>>>(DgridT2, Dgrid2, cols);
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
//        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream2>>>(Dgrid2, DgridT2, cols);
        for(int s=1; s<=log2(cols); s++){
            int m = 1 << s;
            busLoocKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
//            busLoocKer<<<blocks, threadsXBlock, 0, stream2>>>(DgridT2, m, cols);
        }
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
//        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream2>>>(DgridT2, Dgrid2, cols);
        mulKer<<<blocksT, threadsXBlockT, 0, stream>>>(Dgrid, cols, 1.f / (float) size);
//        mulKer<<<blocksT, threadsXBlockT, 0, stream2>>>(Dgrid2, cols, 1.f / (float) size);

        centerKer<<<blocksC, threadsXBlockT, 0, stream>>>(Dgrid, cols);
//        centerKer<<<blocksC, threadsXBlockT, 0, stream2>>>(Dgrid2, cols);
        cudaMemcpyAsync(grid, Dgrid, cuSize, cudaMemcpyDeviceToHost, stream);
//        cudaMemcpyAsync(grid2, Dgrid2, cuSize, cudaMemcpyDeviceToHost, stream2);
        cudaStreamSynchronize(stream);
//        cudaStreamSynchronize(stream2);
    }

    if (saveData){
        // REAL CASE, NOT FOR COMPLEX ORIGINAL IMAGE!!!!
        float *saveFft = new float[size];
        for(int i=0; i<size; i++){
//            saveFft[i] = log(1.f + hypotf(grid[i].real(), grid[i].imag()));
//            saveFft[i] = hypotf(grid[i].real(), grid[i].imag());
            saveFft[i] = grid[i].real();  // real part use this I guess
        }

        // COMPLEX CASE
//        std::complex<float> *saveFft = new std::complex<float>[size];
//        for(int i=0; i<size; i++){
//                saveFft[i] = grid[i];
//        }

        save.open("data/fftCu.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (saveFft), size*sizeof(float) / sizeof(char));
//        save.write(reinterpret_cast<char *> (saveFft), size*sizeof(std::complex<float>) / sizeof(char));
        save.close();
        delete[] saveFft;
    }

	cudaFreeAsync(Dgrid, stream);
//	cudaFreeAsync(Dgrid2, stream2);
	cudaFreeAsync(DgridT, stream);
//	cudaFreeAsync(DgridT2, stream2);
	cudaFreeAsync(grid, stream);
//	cudaFreeAsync(grid2, stream2);
    cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
//    cudaStreamSynchronize(stream2);
//	cudaStreamDestroy(stream2);

    return 0;
}
