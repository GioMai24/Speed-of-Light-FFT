#include <iostream>
#include <fstream>  // file handler
#include <string>
#include <chrono>
#include <complex>
#include <numbers>
#include <ctime>
#include "utils.h"
#include <omp.h>

const std::complex<float> i(0,1);
const int nThreads = std::atoi(getenv("OMP_NUM_THREADS"));  // NOT SAFE IF FORGET TO DEFINE THE ENV
const float pi = std::numbers::pi;



/**
 * The function
 */
float CosCos(const float x, const float y, const float fx, const float fy){
	return cos(2 * pi * fx * x) * cos(2 * pi * fy * y);
}


/**
 * DFT using radix-2 Cooley-Tukey algorithm.
 *
 * @param *x Input 1D array to transform.
 * @param *res Output 1D array to store results.
 * @param N Size of both x and res.
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

int main(){
    using namespace std::chrono;

// frequencies
	const float fx = 0.3;
	const float fy = 0.8;

	// points
	float xMin = 0, xMax = 512;
	float yMin = 0, yMax = 512;
//	 would be nice to throw an error if min > max

	// grid
	const int rows = 2048;
	const int cols = 2048;
	float *grid = new float[rows * cols]; // using doubles must use heap, not stack for large arrays.
	float xStep = (xMax - xMin) / (float) cols;
	float yStep = (yMax - yMin) / (float) rows;

	float xTemp;
	for(int i=0; i<rows; i++){
		xTemp = xMin;
		for(int j=0; j<cols; j++){
			grid[i * cols + j] = CosCos(xTemp, yMin, fx, fy);
			xTemp += xStep;
		}
		yMin += yStep;
	}


	centerSpectrum(grid, rows, cols);

    // fft rows
    std::complex<float> *fft = new std::complex<float>[rows * cols];

    int lCols = log2(cols);
    int revCol[cols];
    for(int j=0; j<cols; j++){
        revCol[j] = revBitOrd(j, lCols);
    }

    steady_clock::time_point start = steady_clock::now();
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            fft[i*cols + revCol[j]] = grid[i*cols + j];
        }
        coolVec(&fft[i * cols], cols);
    }
    steady_clock::time_point stop = steady_clock::now();
    duration<float> time_span = duration_cast<duration<float>>(stop - start);
    std::cout << time_span.count() << std::endl;

    // back to original
    centerSpectrum(grid, rows, cols);


    // transpose
    std::complex<float> *fftT = new std::complex<float>[rows*cols];
    transpose(fft, fftT, rows, cols);

    // fft cols
    int lRows = log2(rows);
    int revRow[rows];
    for(int i=0; i<rows; i++){
        revRow[i] = revBitOrd(i, lRows);
    }

    start = steady_clock::now();
    #pragma omp parallel for
    for(int j=0; j<cols; j++){
        for(int i=0; i<rows; i++){
            fft[j*rows + revRow[i]] = fftT[j*rows + i];
        }
        coolVec(&fft[j * rows], rows);  // overwrites old fft
    }
    stop = steady_clock::now();
    time_span = duration_cast<duration<float>>(stop - start);
    std::cout << time_span.count() << std::endl;

    transpose(fft, fftT, cols, rows);  // overwrites old fftT, now fftT is the DFT of the original dim

    // spectrum
    float *specter = new float[rows*cols];
    spectrum(fftT, specter, rows, cols);
    logSpectrum(specter, rows, cols, 1.f);



//    std::ofstream save;
//	save.open("grid_open.csv");
//	for(int i=0; i<rows; i++){
//		for(int j=0; j<cols; j++){
//			save << grid[i * cols + j];
//			if(j != cols - 1){save << ", ";}
//		}
//		save << '\n';
//	}
//	save.close();
//
//	save.open("fft_open.csv");
//	for(int i=0; i<rows; i++){
//		for(int j=0; j<cols; j++){
//			save << specter[i * cols + j];
//			if(j != cols - 1){save << ", ";}
//		}
//		save << '\n';
//	}
//	save.close();


	delete[] grid;
	delete[] fft;
	delete[] fftT;
	delete[] specter;
    return 0;
}

