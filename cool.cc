#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>
#include <complex>
#include <numbers>
#include <chrono>
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
    for(int s=1; s<=log2(N); s++){
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
    if (argc != 2){
        std::cout << "Something's off dude..." << std::endl;
        return 1;
    }
    using namespace std::chrono;

    const bool saveData = false;

	// save stuff & time
    std::ifstream load;
    std::ofstream save;
    steady_clock::time_point t1, t2;
    duration<double> dtCenter, dtRev, dtFft, dtT, dtGauss, dtIfft;


	// grid
	const std::string sRows = argv[1];
	const int rows = std::stoi(sRows);
	const int cols = rows;
	const size_t size = rows * cols;
//	std::complex<float> *original = new std::complex<float>[size];
	std::complex<float> *grid = new std::complex<float>[size];
	std::complex<float> *gridT = new std::complex<float>[size];
    const int B = 8;
//    int B = atoi(argv[1]);
    const int lCols = log2(cols), lRows = log2(rows);
    int revCol[cols], revRow[rows];
    const float rN = 1.f / (float) size;
    const float rS = 1.f / (2.f * 80.f * 80.f);

    load.open("data/" + sRows + ".bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (grid), nChar);
	load.close();

    t1 = steady_clock::now();
    centerSpectrum(grid, rows, cols);
    t2 = steady_clock::now();
    dtCenter = duration_cast<duration<double>>(t2 - t1);
//	memcpy(original, grid, size * sizeof(std::complex<float>));

    // fft rows
    t1 = steady_clock::now();
    for(int j=0; j<cols; j++){
        revCol[j] = revBitOrd(j, lCols);
    }
    t2 = steady_clock::now();
    dtRev = duration_cast<duration<double>>(t2 - t1);


    t1 = steady_clock::now();
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i * cols + revCol[j]] = grid[i*cols + j];
        }
        coolVec(&gridT[i * cols], cols);
    }
    t2 = steady_clock::now();
    dtFft = duration_cast<duration<double>>(t2 - t1);



    // fft cols
    for(int i=0; i<rows; i++){
        revRow[i] = revBitOrd(i, lRows);
    }

    t1 = steady_clock::now();
    transpose(grid, gridT, rows, cols, B);
    t2 = steady_clock::now();
    dtT = duration_cast<duration<double>>(t2 - t1);

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i*cols + revRow[j]] = grid[i*cols + j];
        }
        coolVec(&gridT[i * cols], cols);
    }

    transpose(grid, gridT, cols, rows, 8);

    // GAUSSIAN FILTERING
    t1 = steady_clock::now();
    for (int i=0; i<rows; i++){
        int x = i - (rows >> 1);
        for (int j=0; j<cols; j++){
            int y = j - (cols >> 1);
            grid[i*cols + j] *= exp(- (float)(x*x + y*y) * rS);
        }
    }
    t2 = steady_clock::now();
    dtGauss = duration_cast<duration<double>>(t2 - t1);

    // INVERSE
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i * cols + revCol[j]] = grid[i*cols + j];
        }
        cevLooc(&gridT[i * cols], cols);
    }

    transpose(grid, gridT, rows, cols, B);


    t1 = steady_clock::now();
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            gridT[i*cols + revRow[j]] = grid[i*cols + j];
        }
        cevLooc(&gridT[i * cols], cols);
    }
    for (int i=0; i<size; i++){
        gridT[i] *= rN;
    }
    t2 = steady_clock::now();
    dtIfft = duration_cast<duration<double>>(t2 - t1);
    transpose(grid, gridT, rows, cols, B);


    centerSpectrum(grid, rows, cols);

    save.open("logs/ser/centerRevFftTGaussIfft.csv", std::ios::app);
    save << sRows << ' '
        << dtCenter.count() << ' '
        << dtRev.count() << ' '
        << dtFft.count() << ' '
        << dtT.count() << ' '
        << dtGauss.count() << ' '
        << dtIfft.count() << std::endl;
    save.close();



//    std::complex<float> diff = 0;
//    for (int i=0; i<size; i++){
//        diff += original[i] - grid[i];
//    }
//    std::cout << "Total difference: " << diff << std::endl;

    // spectrum then log scale//    for (int i=0; i<size; i++){

    if (saveData){
//        float *saveFft = new float[size];
//        for(int i=0; i<size; i++){
//            saveFft[i] = log(1.f + hypotf(grid[i].real(), grid[i].imag()));
//            saveFft[i] = grid[i];
//        }

        save.open("data/fftSer.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (grid), size*sizeof(std::complex<float>) / sizeof(char));
        save.close();
//        delete[] saveFft;
    }

	delete[] grid;
//	delete[] original;
	delete[] gridT;

    return 0;
}
