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
	float xMin = 0, xMax = 256;
	float yMin = 0, yMax = 256;

    std::ofstream save;


	// grid
	const int rows = 1024;
	const int cols = 1024;
	const int size = rows * cols;
	std::complex<float> *grid = new std::complex<float>[size];
	std::complex<float> *gridT = new std::complex<float>[size];
	float xStep = (xMax - xMin) / (float) cols;
	float yStep = (yMax - yMin) / (float) rows;

    steady_clock::time_point t1 = steady_clock::now();
	for(int i=0; i<rows; i++){
		float xTemp = xMin;
		for(int j=0; j<cols; j++){
			grid[i * cols + j] = CosCos(xTemp, yMin, fx, fy);
			xTemp += xStep;
		}
		yMin += yStep;
	}
    steady_clock::time_point t2 = steady_clock::now();
    duration<double> stuff = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Cosing: " << stuff.count() << std::endl;

	save.open("data.csv");
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			save << grid[i * cols + j].real();
			if(j != cols - 1){save << ", ";}
		}
		save << std::endl;
	}
	save.close();

	centerSpectrum(grid, rows, cols);

    // fft rows

    int lCols = log2(cols);
    int revCol[cols];
    for(int j=0; j<cols; j++){
        revCol[j] = revBitOrd(j, lCols);
    }

    steady_clock::time_point start = steady_clock::now();
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            std::complex<float> temp = grid[i * cols + revCol[j]];
            grid[i*cols + revCol[j]] = grid[i*cols + j];
            grid[i*cols + j] = temp;
        }
        coolVec(&grid[i * cols], cols);
    }
    steady_clock::time_point stop = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(stop - start);
    std::cout << "Rows computation: " << time_span.count() << std::endl;


    // fft cols
    // revRowing is useless since it's a square matrix, but you never know...
    int lRows = log2(rows);
    int revRow[rows];
    for(int i=0; i<rows; i++){
        revRow[i] = revBitOrd(i, lRows);
    }

    int B = 1 << 7;
    start = steady_clock::now();
    transpose(grid, gridT, rows, cols, B);
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            std::complex<float> temp = gridT[i*cols + revRow[j]];
            gridT[i*cols + revRow[j]] = gridT[i*cols + j];
            gridT[i*cols + j] = temp;
        }
        coolVec(&gridT[i * cols], cols);
    }
    stop = steady_clock::now();
    time_span = duration_cast<duration<double>>(stop - start);
    std::cout << "transposed blocked 128 one arr: " << time_span.count() << std::endl;
    transpose(gridT, grid, cols, rows, B);


    // spectrum then log scale
    for(int i=0; i<size; i++){
        grid[i] = log(1.f + abs(grid[i]));
    }

	save.open("fft.csv");
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

	delete[] grid;
	delete[] gridT;

    return 0;
}
