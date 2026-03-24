#include <iostream>
#include <fstream>  // file handler
#include <string>
#include <chrono>
#include <complex>
#include <numbers>
#include <ctime>
#include "utils.h"
#include <omp.h>

const std::complex<double> i(0,1);
const int nThreads = std::atoi(getenv("OMP_NUM_THREADS"));  // NOT SAFE IF FORGET TO DEFINE THE ENV
const double pi = std::numbers::pi;



/**
 * The function
 */
double CosCos(const double x, const double y, const double fx, const double fy){
	return cos(2 * pi * fx * x) * cos(2 * pi * fy * y);
}


/**
 * DFT using radix-2 Cooley-Tukey algorithm.
 *
 * @param *x Input 1D array to transform.
 * @param *res Output 1D array to store results.
 * @param N Size of both x and res.
 */
void coolVec(std::complex<double> *res, int N){
    int lN = log2(N);
    std::complex<double> common = - 2 * pi * i;  // might have sign problem...
    for(int s=1; s<log2(N)+1; s++){
        int m = 1 << s;
        std::complex<double> wm = exp(common / (double) m);
        for(int k=0; k<N; k+=m){
            std::complex<double> w = 1;
            for(int j=0; j<m/2; j++){
                std::complex<double> t = w * res[k+j+m/2];
                std::complex<double> u = res[k+j];
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
	const double fx = 0.3;
	const double fy = 0.6;

	// points
	double xMin = 0, xMax = 2048;
	double yMin = 0, yMax = 2048;
//	 would be nice to throw an error if min > max

	// grid
	const int rows = 8192;
	const int cols = 8192;
	double *grid = new double[rows * cols]; // using doubles must use heap, not stack for large arrays.
	double xStep = (xMax - xMin) / (double) cols;
	double yStep = (yMax - yMin) / (double) rows;

	double xTemp;
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
    std::complex<double> *fft = new std::complex<double>[rows * cols];

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
    duration<double> time_span = duration_cast<duration<double>>(stop - start);
    std::cout << time_span.count() << std::endl;

    // back to original
    centerSpectrum(grid, rows, cols);


    // transpose
    std::complex<double> *fftT = new std::complex<double>[rows*cols];
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
    time_span = duration_cast<duration<double>>(stop - start);
    std::cout << time_span.count() << std::endl;

    transpose(fft, fftT, cols, rows);  // overwrites old fftT, now fftT is the DFT of the original dim
//    printArray(fftT, rows, cols);

    // spectrum
    double *specter = new double[rows*cols];
    spectrum(fftT, specter, rows, cols);
    logSpectrum(specter, rows, cols, 1.);



//    clock_t start = clock();
//    coolVec(x, res, cols);
//    clock_t stop = clock();
//    std::cout << (double)(stop - start) / CLOCKS_PER_SEC << std::endl;
//
//    std::ofstream saveGrid;
//	saveGrid.open("grid.csv");
//	for(int i=0; i<rows; i++){
//		for(int j=0; j<cols; j++){
//			saveGrid << grid[i * cols + j];
//			if(j != cols - 1){saveGrid << ", ";}
//		}
//		saveGrid << '\n';
//	}
//	saveGrid.close();
//
//	std::ofstream saveFft;
//	saveFft.open("fft.csv");
//	for(int i=0; i<rows; i++){
//		for(int j=0; j<cols; j++){
//			saveFft << specter[i * cols + j];
//			if(j != cols - 1){saveFft << ", ";}
//		}
//		saveFft << '\n';
//	}
//	saveFft.close();

	delete[] grid;
	delete[] fft;
	delete[] fftT;
	delete[] specter;
    return 0;
}
/*
for(int k=0; k<N2; k++){
        std::complex<double> E(0,0);
        std::complex<double> O(0,0);
        for(int m=0; m<N2; m++){
            std::complex<double> twiddle = exp(-(4 * pi * m * k / N) * i);
            E += x[2*m] * twiddle;
            O += x[2*m+1] * twiddle;
        }
        std::complex<double> expO = exp(-(2 * pi * k / N) * i);
        res[k] = E + expO * O;
        res[k+N2] = E - expO * O;
    }
*/
