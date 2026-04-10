#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <complex>
#include <numbers>
#include <chrono>
#include <omp.h>
#include "utilsMP.h"

const std::complex<float> i(0,1);
const float pi = std::numbers::pi;
const int nThreads = std::atoi(getenv("OMP_NUM_THREADS"));


/**
 * DFT using radix-2 Cooley-Tukey algorithm.
 *
 * @param *res Output 1D array to store results.
 * @param N Size of res.
 */
void coolVec(std::complex<float> *res, int N){
    int lN = log2(N);
    std::complex<float> common = - 2 * pi * i;  // might have sign problem...
    for(int s=1; s<log2(N)+1; s++){
        int m = 1 << s;
        std::complex<float> wm = exp(common / (float) m);
        for(int k=0; k<N; k+=m){
            std::complex<float> w = 1;
            for(int j=0; j<m/2; j++){
                std::complex<float> t = w * res[k+j+m/2];
                std::complex<float> u = res[k+j];
                res[k+j] = u + t;
                res[k+j+m/2] = u - t;
                w *= wm;
            }
        }
    }
}



int main(int argc, char **argv){
//    if (argc==1){
//        std::cout << "Missing block size" << std::endl;
//        return 1;
//    }
    using namespace std::chrono;

    bool saveData = false;

	// save stuff & time
    std::ifstream load;
    std::ofstream save;
    steady_clock::time_point t1;
    steady_clock::time_point t2;
    duration<double> dt;


	// grid
	const int rows = 16384;
	const int cols = rows;
	const size_t size = rows * cols;
	std::complex<float> *grid = new std::complex<float>[size];
	std::complex<float> *gridT = new std::complex<float>[size];
    int B = 8;
//    int B = atoi(argv[1]);

    load.open("data/16384.bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (grid), nChar);
	load.close();

    if (saveData){
        centerSpectrum(grid, rows, cols);
    }

    // fft rows
    // ordering
    int lCols = log2(cols);
    int revCol[cols];
    t1 = steady_clock::now();
    #pragma omp parallel for
    for(int j=0; j<cols; j++){
        revCol[j] = revBitOrd(j, lCols);
    }

    // actual fft
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i * cols + revCol[j]] = grid[i*cols + j];
        }
        coolVec(&gridT[i * cols], cols);
    }


    // fft cols
    // revRowing is useless since it's a square matrix, but you never know...
    int lRows = log2(rows);
    int revRow[rows];
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        revRow[i] = revBitOrd(i, lRows);
    }

    transpose(gridT, grid, rows, cols, B);
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i*cols + revRow[j]] = grid[i*cols + j];
        }
        coolVec(&gridT[i * cols], cols);
    }

    transpose(gridT, grid, cols, rows, B);
    t2 = steady_clock::now();
    dt = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Time: " << dt.count() << std::endl;

    // spectrum then log scale
    if (saveData){
        float *saveFft = new float[size];
        #pragma omp parallel for
        for(int i=0; i<size; i++){
            saveFft[i] = log(1.f + hypotf(grid[i].real(), grid[i].imag()));
        }

        save.open("data/fftOmp.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (saveFft), size*sizeof(float) / sizeof(char));
        save.close();
        delete[] saveFft;
    }

	delete[] grid;
	delete[] gridT;

    return 0;
}
