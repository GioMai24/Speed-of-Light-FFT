#include <iostream>
#include <fstream>  // file handler
#include <string>
#include <cmath>
#include <complex>
#include <numbers>
#include <chrono>
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


int main(){
    using namespace std::chrono;
    // frequencies
	const float fx = 0.3;
	const float fy = 0.8;

	// points
	float xMin = 0, xMax = 512;
	float yMin = 0, yMax = 512;

    std::ofstream save;


	// grid
	const int rows = 2048;
	const int cols = 2048;
	const int size = rows * cols;
	float *grid = new float[size];
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

//	save.open("data.csv");
//	for(int i=0; i<rows; i++){
//		for(int j=0; j<cols; j++){
//			save << grid[i * cols + j];
//			if(j != cols - 1){save << ", ";}
//		}
//		save << std::endl;
//	}
//	save.close();

	centerSpectrum(grid, rows, cols);

    // fft rows
    std::complex<float> *fft = new std::complex<float>[size];

    int lCols = log2(cols);
    int revCol[cols];
    for(int j=0; j<cols; j++){
        revCol[j] = revBitOrd(j, lCols);
    }

    steady_clock::time_point start = steady_clock::now();
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            fft[i*cols + revCol[j]] = grid[i*cols + j];
        }
        coolVec(&fft[i * cols], cols);
    }
    steady_clock::time_point stop = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(stop - start);
    std::cout << "Rows computation: " << time_span.count() << std::endl;


    // fft cols
    std::complex<float> *fft2 = new std::complex<float>[size];
    // revRowing is useless since it's a square matrix, but you never know...
    int lRows = log2(rows);
    int revRow[rows];
    for(int i=0; i<rows; i++){
        revRow[i] = revBitOrd(i, lRows);
    }


    // coolVec inside the loop
    start = steady_clock::now();
    for(int j=0; j<cols; j++){
        for(int i=0; i<rows; i++){
            fft2[j*rows + revRow[i]] = fft[i*cols+j];
        }
        coolVec(&fft2[j * rows], rows);
    }
    stop = steady_clock::now();
    time_span = duration_cast<duration<double>>(stop - start);
    std::cout << "Cols comp, single loop: " << time_span.count() << std::endl;

    // spectrum then log scale
    float *specter = new float[size];
    for(int i=0; i<size; i++){
        specter[i] = 5.f * log(1.f + abs(fft2[i]));
    }

//	save.open("fftAngle.csv");
//	for(int i=0; i<rows; i++){
//		for(int j=0; j<cols; j++){
//			save << specter[j * rows + i];
//			if(j != rows - 1){
//                save << ", ";
//            }
//		}
//		save << std::endl;
//	}
//	save.close();

	delete[] grid;
	delete[] fft;
	delete[] fft2;
	delete[] specter;

	std::cout << std::endl;

    return 0;
}
