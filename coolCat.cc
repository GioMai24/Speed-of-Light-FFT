#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>
#include <complex>
#include <numbers>
#include <chrono>
#include <cstdint>
#include "utils.h"


/**
 * DFT using radix-2 Cooley-Tukey algorithm.
 *
 * @param *res Output 1D array to store results.
 * @param N Size of res.
 */
void coolVec(std::complex<float> *res, int N){
    for(int s=1; s<=log2(N); s++){
        int m = 1 << s;
        std::complex<float> wm = std::polar(1.f, -2 * std::numbers::pi_v<float> / (float) m);
        for(int k=0; k<N; k+=m){
            std::complex<float> w = 1;
            for(int j=0; j<(m >> 1); j++){
                std::complex<float> t = w * res[k+j+(m >> 1)];
                std::complex<float> u = res[k+j];
                res[k+j] = u + t;
                res[k+j+(m >> 1)] = u - t;
                w *= wm;
            }
        }
    }
}



void coolDVec(std::complex<float> *res, int N){
    int lN = log2(N);
    for(int s=1; s<=lN; s++){
        int m = 1 << s;
        for(int l=0; l<(N >> 1); l++){
            int j = l & ((m >> 1) - 1);  // modulo
            int k = (2 * l >> (int)log2(m)) * m;
            std::complex<float> w = std::polar(1.f, -2 * j * std::numbers::pi_v<float> / (float) m);

            std::complex<float> t = w * res[k + j + (m >> 1)];
            std::complex<float> u = res[k + j];
            res[k + j] = u + t;
            res[k + j + (m >> 1)] = u - t;
        }
    }
}


void cevLooc(std::complex<float> *res, int N){
    for(int s=1; s<=log2(N); s++){
        int m = 1 << s;
        std::complex<float> wm = std::polar(1.f, 2 * std::numbers::pi_v<float> / (float) m);
        for(int k=0; k<N; k+=m){
            std::complex<float> w = 1;
            for(int j=0; j<(m >> 1); j++){
                std::complex<float> t = w * res[k+j+(m >> 1)];
                std::complex<float> u = res[k+j];
                res[k+j] = u + t;
                res[k+j+(m >> 1)] = u - t;
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

	// save stuff & time
    bool saveData = true;
    std::ifstream load;
    std::ofstream save;
    steady_clock::time_point t1;
    steady_clock::time_point t2;
    duration<double> dt;


	// grid
	const int rows = 1024;
	const int cols = rows;
	const size_t size = rows * cols;
	uint8_t *img = new uint8_t[512*512];

	std::complex<float> *grid = new std::complex<float>[size];
	std::complex<float> *original = new std::complex<float>[size];
	std::complex<float> *gridT = new std::complex<float>[size];

	load.open("data/cats/cat512.bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (img), nChar);
	load.close();

    int B = 8;
//    int B = atoi(argv[1]);

    // TRANSLATE IN COMPLEX + PADDING
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            grid[i * cols + j] = (i < 512 && j < 512) ? img[i * 512 + j] : 0;
//            grid[i * cols + j] = img[i * 512 + j];
        }
    }

    delete[] img;

	if (saveData){
        centerSpectrum(grid, rows, cols);
	}
	memcpy(original, grid, size * sizeof(std::complex<float>));

    // fft rows
    // ordering
    int lCols = log2(cols);
    int revCol[cols];
    t1 = steady_clock::now();
    for(int j=0; j<cols; j++){
        revCol[j] = revBitOrd(j, lCols);
    }

    // actual fft
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
    for(int i=0; i<rows; i++){
        revRow[i] = revBitOrd(i, lRows);
    }

    transpose(grid, gridT, rows, cols, B);
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i*cols + revRow[j]] = grid[i*cols + j];
        }
        coolVec(&gridT[i * cols], cols);
    }

    transpose(grid, gridT, cols, rows, 8);
    t2 = steady_clock::now();
    dt = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Serial: " << dt.count() << " s" << std::endl;


    float *saveFft = new float[size];
    for(int i=0; i<size; i++){
        saveFft[i] = log(1.f + hypotf(grid[i].real(), grid[i].imag()));
    }
    save.open("data/cats/out/furiousCat.bin", std::ios::binary);
    save.write(reinterpret_cast<char *> (saveFft), size*sizeof(float) / sizeof(char));
    save.close();
    delete[] saveFft;


    // GAUSSIAN FILTERING
    for (int i=0; i<rows; i++){
        int x = i - (rows >> 1);
        for (int j=0; j<cols; j++){
            int y = j - (cols >> 1);
            grid[i*cols + j] *= exp(- (float)(x*x + y*y) / (2 * 20 * 20));
        }
    }

    // INVERSE
    float iN = 1.f / (float) size;

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i * cols + revCol[j]] = grid[i*cols + j];
        }
        cevLooc(&gridT[i * cols], cols);
    }

    transpose(grid, gridT, rows, cols, B);

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i*cols + revRow[j]] = grid[i*cols + j];
        }
        cevLooc(&gridT[i * cols], cols);
    }
    for (int i=0; i<size; i++){
        gridT[i] *= iN;
    }
    transpose(grid, gridT, rows, cols, B);


    std::complex<float> diff = 0;
    for (int i=0; i<size; i++){
        diff += original[i] - grid[i];
    }
    std::cout << "Total difference: " << diff << std::endl;

    // spectrum then log scale

    if (saveData){
        float *saveFft = new float[512*512];
        // log specter...
        for(int i=0; i<512; i++){
            for(int j=0; j<512; j++){
//                saveFft[i] = log(1.f + hypotf(grid[i].real(), grid[i].imag()));  // log specter
                saveFft[i*512+j] = hypotf(grid[i*cols+j].real(), grid[i*cols+j].imag());  // plain spectrum
//                saveFft[i] = grid[i].real();  // real part
            }
        }

        save.open("data/cats/out/blurCat.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (saveFft), 512*512*sizeof(float) / sizeof(char));
        save.close();

        delete[] saveFft;
    }

	delete[] grid;
	delete[] original;
	delete[] gridT;

    return 0;
}
