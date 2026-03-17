#include <iostream>
#include <fstream>  // file handler
#include <cmath>
#include <complex>
#include <numbers>  // requires -std=c++20
#include <ctime>
#include "utils.h"

/**
 * std::numbers::pi is a double by default. To avoid casting I set everything to double.
 */

const std::complex<double> i(0,1);


/**
 * The function
 */
double CosCos(const double x, const double y, const double fx, const double fy){
	return cos(2 * std::numbers::pi * fx * x) * cos(2 * std::numbers::pi * fy * y);
}


/**
 * DFT using radix-2 Cooley-Tukey algorithm.
 *
 * @param *x Input 1D array to transform.
 * @param *res Output 1D array to store results.
 * @param N Size of both x and res.
 */
template<typename T>
void coolVec(T *x, std::complex<double> *res, int N){
    int N2 = N/2;
    /*
    // computation for k=0. test if the if clause makes it faster...
    double sum = 0;
    for(int m=0; m<N; m++){
        sum += x[m];
    }
    std::cout << sum << std::endl;
    */
    for(int k=0; k<N2; k++){
        std::complex<double> E(0,0);
        std::complex<double> O(0,0);
        for(int m=0; m<N2; m++){
            std::complex<double> twiddle = exp(-(4 * std::numbers::pi * m * k / N) * i);
            E += x[2*m] * twiddle;
            O += x[2*m+1] * twiddle;
        }
        std::complex<double> expO = exp(-(2 * std::numbers::pi * k / N) * i);
        res[k] = E + expO * O;
        res[k+N2] = E - expO * O;
    }
}


int main(){
    // frequencies
	const double fx = 0.3;
	const double fy = 0.6;

	// points
	double xMin = 0, xMax = 15;
	double yMin = 0, yMax = 15;
//	 would be nice to throw an error if min > max

	// grid
	const int rows = 512;
	const int cols = 512;
	double grid[rows * cols];
	double xStep = (xMax - xMin) / (double) cols;  // these lines are useless?
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
    std::complex<double> fft[rows * cols];
    for(int i=0; i<rows; i++){
        coolVec(&grid[i * cols], &fft[i * cols], cols);
    }
    centerSpectrum(grid, rows, cols);


    // transpose
    std::complex<double> fftT[rows*cols];
    transpose(fft, fftT, rows, cols);
    // fft cols
    for(int j=0; j<cols; j++){
        coolVec(&fftT[j * rows], &fft[j * rows], rows);  // overwrites old fft
    }
    transpose(fft, fftT, cols, rows);  // overwrites old fftT, now fftT is the DFT of the original dim
//    printArray(fftT, rows, cols);

    // spectrum
    double specter[rows*cols];
    spectrum(fftT, specter, rows, cols);
    logSpectrum(specter, rows, cols, 1.);



//    clock_t start = clock();
//    coolVec(x, res, cols);
//    clock_t stop = clock();
//    std::cout << (double)(stop - start) / CLOCKS_PER_SEC << std::endl;
//
    std::ofstream saveGrid;
	saveGrid.open("grid.csv");
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			saveGrid << grid[i * cols + j];
			if(j != cols - 1){saveGrid << ", ";}
		}
		saveGrid << '\n';
	}
	saveGrid.close();

	std::ofstream saveFft;
	saveFft.open("fft.csv");
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			saveFft << specter[i * cols + j];
			if(j != cols - 1){saveFft << ", ";}
		}
		saveFft << '\n';
	}
	saveFft.close();
    return 0;
}
