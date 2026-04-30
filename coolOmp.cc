#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <complex>
#include <numbers>
#include <chrono>
#include "utilsMP.h"

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
            for(int j=0; j<(m>>1); j++){
                std::complex<float> t = w * res[k+j+(m>>1)];
                std::complex<float> u = res[k+j];
                res[k+j] = u + t;
                res[k+j+(m>>1)] = u - t;
                w *= wm;
            }
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

    bool saveData = false;

	// IO stuff & time
    std::ifstream load;
    std::ofstream save;
    steady_clock::time_point t1, t2;
    duration<double> dt;


	// grid (unfortunate notation for data vs cuGrid...)
	std::string sRows = argv[1];
	const int rows = std::stoi(sRows);
	const int cols = rows;
	const size_t size = rows * cols;
	std::complex<float> *grid = new std::complex<float>[size];
	std::complex<float> *gridT = new std::complex<float>[size];
    int B = 64;
    int lCols = log2(cols), lRows=log2(rows);
    int revCol[cols], revRow[rows];
    float rN = 1.f / (float) size;
    float rS = 1.f / (2.f * 80.f * 80.f);

    for (int counter=0; counter<100; counter++)
    {
        load.open("data/" + sRows + ".bin", std::ios::binary | std::ios::ate);
    //    load.open("data/cats/cut4K2048.bin", std::ios::binary | std::ios::ate);
        std::streamsize nChar = load.tellg();
        load.seekg(0);
        load.read(reinterpret_cast<char *> (grid), nChar);
        load.close();


//        t1 = steady_clock::now();
        centerSpectrum(grid, rows, cols);

        #pragma omp parallel
        {
            // FFT ROWS
            #pragma omp for
            for(int j=0; j<cols; j++){
                revCol[j] = revBitOrd(j, lCols);
            }
            #pragma omp for schedule(static, 1)
            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){
                    gridT[i * cols + revCol[j]] = grid[i*cols + j];
                }
                coolVec(&gridT[i * cols], cols);
            }
    // TO TIMEEEEEEE
            // FFT COLS
            #pragma omp for
            for(int i=0; i<rows; i++){
                revRow[i] = revBitOrd(i, lRows);
            }
        }
        transpose(gridT, grid, rows, cols, B);

        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 1)
            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){
                    gridT[i*cols + revRow[j]] = grid[i*cols + j];
                }
                coolVec(&gridT[i * cols], cols);
            }
        }
        transpose(gridT, grid, cols, rows, B);
        #pragma omp parallel
        {
            // GAUSSIAN FILTERING
            #pragma omp for
            for (int i=0; i<rows; i++){
                int x = i - (rows >> 1);
                for (int j=0; j<cols; j++){
                    int y = j - (cols >> 1);
                    grid[i*cols + j] *= exp(- (float)(x*x + y*y) * rS);
                }
            }
    // TO TIMEEEEEE
            // IFFT ROWS
            #pragma omp for
            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){
                    gridT[i * cols + revCol[j]] = grid[i*cols + j];
                }
                cevLooc(&gridT[i * cols], cols);
            }
        }
        // IFFT COLS
        transpose(gridT, grid, rows, cols, B);

        #pragma omp parallel
        {
            #pragma omp for
            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){
                    gridT[i*cols + revRow[j]] = grid[i*cols + j];
                }
                cevLooc(&gridT[i * cols], cols);
            }

            #pragma omp for
            for (int i=0; i<size; i++){
                gridT[i] *= rN;
            }
        }
        transpose(gridT, grid, rows, cols, B);


        // spectrum then log scale
        centerSpectrum(grid, rows, cols);  // put complex back lol
    }
//    t2 = steady_clock::now();
//    dt = duration_cast<duration<double>>(t2 - t1);

//    save.open("logs/OMP/N" + sRows + "Th" + std::getenv("OMP_NUM_THREADS") + "RevFftstatandDynTGaussIfft.txt", std::ios::app);
//    save.open("logs/OMP/pipeline.csv", std::ios::app);
//    save << std::getenv("OMP_NUM_THREADS") << ' '
//		<< sRows << ' '
//        << dt.count() << std::endl;
//    save.close();

    if (saveData){
        // REAL CASE, NOT FOR COMPLEX ORIGINAL IMAGE!!!!
        float *saveFft = new float[size];
        for(int i=0; i<size; i++){
//            saveFft[i] = log(1.f + hypotf(grid[i].real(), grid[i].imag()));
//            saveFft[i] = hypotf(grid[i].real(), grid[i].imag());
            saveFft[i] = grid[i].real();  // real part use this I guess
        }
        // COMPLEX CASE
//        std::complex<float> *saveFft = new std::complex<float>[size];
//        for(int i=0; i<size; i++){
//                saveFft[i] = grid[i];
//        }

        save.open("data/fftOmp.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (saveFft), size*sizeof(float) / sizeof(char));
//        save.write(reinterpret_cast<char *> (saveFft), size*sizeof(std::complex<float>) / sizeof(char));
        save.close();
        delete[] saveFft;
    }

	delete[] grid;
	delete[] gridT;

    return 0;
}
