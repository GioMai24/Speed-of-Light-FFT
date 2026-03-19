#include <iostream>
#include <fstream>  // file handler
#include <string>
#include <cmath>
#include <complex>
#include <numbers>
#include <ctime>
#include "utils.h"

const std::complex<float> i(0,1);
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
template<typename T>
void coolVec(T *x, std::complex<float> *res, int N){
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
    // frequencies
	const float fx = 0.3;
	const float fy = 0.6;

	// points
	float xMin = 0, xMax = 16;
	float yMin = 0, yMax = 16;
//	 would be nice to throw an error if min > max

	// grid
	const int rows = 2048;
	const int cols = 2048;
	float *grid = new float[rows * cols]; // using floats must use heap, not stack for large arrays.
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

//    clock_t start = clock();
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            fft[i*cols + revCol[j]] = grid[i*cols + j];
        }
        coolVec(&grid[i * cols], &fft[i * cols], cols);
    }
//    clock_t stop = clock();
//    std::cout << (stop - start) << std::endl;
    // back to original grid is not used if not for saving
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

//    start = clock();
    for(int j=0; j<cols; j++){
        for(int i=0; i<rows; i++){
            fft[j*rows + revRow[i]] = fftT[j*rows + i];
        }
        coolVec(&fftT[j * rows], &fft[j * rows], rows);  // overwrites old fft
    }
//    stop = clock();
//    std::cout << (stop - start) << std::endl;

    transpose(fft, fftT, cols, rows);  // overwrites old fftT, now fftT is the DFT of the original dim
//    printArray(fftT, rows, cols);

    // spectrum
    float *specter = new float[rows*cols];
    spectrum(fftT, specter, rows, cols);
    logSpectrum(specter, rows, cols, 1.f);



//    clock_t start = clock();
//    coolVec(x, res, cols);
//    clock_t stop = clock();
//    std::cout << (float)(stop - start) / CLOCKS_PER_SEC << std::endl;
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
