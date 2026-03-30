#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <complex>
#include <numbers>
#include <chrono>
#include <omp.h>
#include "utils.h"

const std::complex<float> i(0,1);
const float pi = std::numbers::pi;
const int nThreads = std::atoi(getenv("OMP_NUM_THREADS"));


/**
 * The function
 */
float CosCos(const float x, const float y, const float fx, const float fy){
	return cos(2 * pi * fx * x) * cos(2 * pi * fy * y);
}


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
	std::complex<float> *grid = new std::complex<float>[size];
	std::complex<float> *gridT = new std::complex<float>[size];
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
        save.open("dataOpen2.csv");
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
    #pragma omp parallel for
    for(int j=0; j<cols; j++){
        revCol[j] = revBitOrd(j, lCols);
    }

    // actual fft
    t1 = steady_clock::now();
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i * cols + revCol[j]] = grid[i*cols + j];
        }
        coolVec(&gridT[i * cols], cols);
    }
    t2 = steady_clock::now();
    dt = duration_cast<duration<double>>(t2 - t1);
//    std::cout << "Rows computation: " << dt.count() << std::endl;


    // fft cols
    // revRowing is useless since it's a square matrix, but you never know...
    int lRows = log2(rows);
    int revRow[rows];
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        revRow[i] = revBitOrd(i, lRows);
    }

    t1 = steady_clock::now();
    transpose(gridT, grid, rows, cols, B);
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i*cols + revRow[j]] = grid[i*cols + j];
        }
        coolVec(&gridT[i * cols], cols);
    }
    t2 = steady_clock::now();
    dt = duration_cast<duration<double>>(t2 - t1);
//    std::cout << "transposed blocked " << B << " one arr: " << dt.count() << std::endl;

    t1 = steady_clock::now();
    transpose(gridT, grid, cols, rows, 8);
    t2 = steady_clock::now();
    dt = duration_cast<duration<double>>(t2 - t1);
    std::cout << "blocked 8 " << dt.count() << std::endl;

    if (saveData){
        // spectrum then log scale
        #pragma omp parallel for
        for(int i=0; i<size; i++){
            grid[i] = log(1.f + abs(grid[i]));
        }

        save.open("fftOpen2.csv");
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

    return 0;
}
