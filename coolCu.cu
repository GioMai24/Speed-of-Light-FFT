#include <iostream>
#include <fstream>
#include <string>
#include <cuda/cmath>
#include <cuda/std/complex>
#include <cuda/std/numbers>
#include <chrono>
#include "utils.h"

namespace cst{
    const cuda::std::complex<float> i(0,1);
    const float pi = cuda::std::numbers::pi;
}

/**
 * The function
 */
float CosCos(const float x, const float y, const float fx, const float fy){
	return cos(2 * cst::pi * fx * x) * cos(2 * cst::pi * fy * y);
}


///**
// * Modify index order based on bit representation reversal.
// *
// * Example: x=3 -> 011 -> 110 -> 6
// */
//int revBitOrd(int x, int lN){
//    int n = 0;
//    for(int i=0; i<lN; i++){
//        n <<= 1;
//        n |= (x & 1);
//        x >>= 1;
//    }
//    return n;
//__global__ void revBitOrdKer(cuda::std::complex *in, cuda::std::complex *out, const int rows, const int cols){
//    int tId = blockDim.x * blockIdx.x + threadIdx.x;
//    if (tId < rows * cols){
//        const int size = rows * cols;
//        const int lCols = cuda::ilog2(cols);
//        int rev = tId % cols;
//        int ori = rev;
//        int n = 0;
//        for(int i=0; i<lCols; i++){
//            n <<= 1;
//            n |= (rev & 1);
//            rev >>= 1;
//        }
//        int diff = ori - n;
//        out[tId] = in[tId - diff];
//    }
//}
//
//    for(int i=0; i<rows; i++){
//        for(int j=0; j<cols; j++){
//            gridT[i * cols + revCol[j]] = grid[i*cols + j];
//        }
//        coolVec(&gridT[i * cols], cols);
//    }


void coolVec(cuda::std::complex<float> *res, int N){
    cuda::std::complex<float> common = - 2 * cst::pi * cst::i;  // might have sign problem...
    for(int s=1; s<log2(N)+1; s++){
        int m = 1 << s;
        cuda::std::complex<float> wm = exp(common / (float) m);
        for(int k=0; k<N; k+=m){
            cuda::std::complex<float> w = 1;
            for(int j=0; j<m/2; j++){
                cuda::std::complex<float> t = w * res[k+j+m/2];
                cuda::std::complex<float> u = res[k+j];
                res[k+j] = u + t;
                res[k+j+m/2] = u - t;
                w *= wm;
            }
        }
    }
}


/**
 * DFT using radix-2 Cooley-Tukey algorithm applied along res rows.
 *
 * @param *res Output 1D array to store results.
 * @param rows Rows to use.
 * @param cols Cols of res to calculate starting idx.
 */
__global__ void coolKer(cuda::std::complex<float> *res, const int rows, const int cols){
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    if (tId < rows){
        cuda::std::complex<float> common = - 2 * cst::pi * cst::i;
        for(int s=1; s<cuda::ilog2(cols)+1; s++){
            int m = 1 << s;
            cuda::std::complex<float> wm = exp(common / (float) m);
            for(int k=0; k<cols; k+=m){
                cuda::std::complex<float> w = 1;
                for(int j=0; j<m/2; j++){
                    cuda::std::complex<float> t = w * res[tId * cols + k+j+m/2];
                    cuda::std::complex<float> u = res[tId * cols + k+j];
                    res[tId * cols + k+j] = u + t;
                    res[tId * cols + k+j+m/2] = u - t;
                    w *= wm;
                }
            }
        }
    }
}


/**
 * Set <<<(grid.x, rows), threads>>> where grid.x * threads = cols/2
 */
__global__ void coolSubKer(cuda::std::complex<float> *res, const int m, const int cols){
    int xId = blockDim.x * blockIdx.x + threadIdx.x;
    if (xId < cols/2){
        int j = xId % (m/2);
        int k = 2 * xId / m;
        k *= m;
        cuda::std::complex<float> w = pow(exp(-2 * cst::pi * cst::i / (float) m), j);

        cuda::std::complex<float> t = w * res[k + j + m/2 + blockIdx.y * cols];
        cuda::std::complex<float> u = res[k + j + blockIdx.y * cols];
        res[k + j + blockIdx.y * cols] = u + t;
        res[k + j + m/2 + blockIdx.y * cols] = u - t;
    }

}


int main(int argc, char **argv){
//    if (argc==1){
//        std::cout << "Missing block size" << std::endl;
//        return 1;
//    }
    using namespace std::chrono;

    bool saveData = true;

    // frequencies
	const float fx = 0.3;
	const float fy = 0.8;

	// points
	float xMin = 0, xMax = 512;
	float yMin = 0, yMax = 512;

	// save stuff & time
    std::ofstream save;
    steady_clock::time_point t1;
    steady_clock::time_point t2;
    duration<double> dt;


	// grid
	const int rows = 1024;
	const int cols = 1024;
	const int size = rows * cols;
	const int cuSize = size * sizeof(cuda::std::complex<float>);
	cuda::std::complex<float> *grid = new cuda::std::complex<float>[size];
	cuda::std::complex<float> *Dgrid = nullptr;
	cuda::std::complex<float> *gridT = new cuda::std::complex<float>[size];
	cuda::std::complex<float> *DgridT = nullptr;

	cudaMalloc(&Dgrid, cuSize);
	cudaMalloc(&DgridT, cuSize);

	float xStep = (xMax - xMin) / (float) cols;
	float yStep = (yMax - yMin) / (float) rows;
    int B = 8;
//    int B = atoi(argv[1]);


	// data gen
	for(int i=0; i<rows; i++){
		float xTemp = xMin;
		for(int j=0; j<cols; j++){
			grid[i * cols + j] = CosCos(xTemp, yMin, fx, fy);
			xTemp += xStep;
		}
		yMin += yStep;
	}

	// data save
	if (saveData){
        save.open("dataCu.csv");
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                save << grid[i * cols + j].real();
                if(j != cols - 1){save << ", ";}
            }
            save << std::endl;
        }
        save.close();

        centerSpectrum(grid, rows, cols);
	}


    // fft rows
    // ordering
    int lCols = log2(cols);
    int revCol[cols];
    for(int j=0; j<cols; j++){
        revCol[j] = revBitOrd(j, lCols);
    }

    // actual fft
//    t1 = steady_clock::now();
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i * cols + revCol[j]] = grid[i*cols + j];
        }
//        coolVec(&gridT[i * cols], cols);
    }
	cudaMemcpy(Dgrid, gridT, cuSize, cudaMemcpyHostToDevice);
//    coolKer<<<16, 64>>>(Dgrid, rows, cols);
//    cudaDeviceSynchronize();
//    t2 = steady_clock::now();
//    cudaMemcpy(gridT, Dgrid, cuSize, cudaMemcpyDeviceToHost);
//    dt = duration_cast<duration<double>>(t2 - t1);
//    std::cout << "Rows cuComputation: " << dt.count() << std::endl;

// NEW ALTERNATIVE
    dim3 gridThreads(16, 1024);
    t1 = steady_clock::now();
    for(int s=1; s<=log2(cols); s++){
        int m = 1 << s;
        coolSubKer<<<gridThreads, 32>>>(Dgrid, m, cols);
        cudaDeviceSynchronize();
    }
    t2 = steady_clock::now();
    cudaMemcpy(gridT, Dgrid, cuSize, cudaMemcpyDeviceToHost);
    dt = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Rows cuComputation: " << dt.count() << std::endl;


    // fft cols
    // revRowing is useless since it's a square matrix, but you never know...
    int lRows = log2(rows);
    int revRow[rows];
    for(int i=0; i<rows; i++){
        revRow[i] = revBitOrd(i, lRows);
    }

    t1 = steady_clock::now();
    transpose(gridT, grid, rows, cols, B);
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i*cols + revRow[j]] = grid[i*cols + j];
        }
        coolVec(&gridT[i * cols], cols);
    }
    t2 = steady_clock::now();
    dt = duration_cast<duration<double>>(t2 - t1);
//    std::cout << "transposed blocked " << B << " one arr: " << dt.count() << std::endl;

    transpose(gridT, grid, cols, rows, 8);

    // spectrum then log scale
    if (saveData){
        for(int i=0; i<size; i++){
            grid[i] = log(1.f + abs(grid[i]));
        }

        save.open("fftCu.csv");
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                save << grid[i * cols + j].real();
                if(j != cols - 1){
                    save << ", ";
                }
            }
            save << std::endl;
        }
        save.close();
    }

	delete[] grid;
	delete[] gridT;
	cudaFree(Dgrid);
	cudaFree(DgridT);

    return 0;
}
