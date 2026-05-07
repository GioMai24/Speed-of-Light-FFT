#include "utilsMP.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <chrono>

/** @file
 * @brief CPU parallel DFT implementation.
 *
 * Compute 100 "images". SET OMP_NUM_THREADS and scheduling?.
 */



int main(int argc, char **argv){
    if (argc != 2){
        std::cout << "Give me some dimensions!" << std::endl;
        return 1;
    }
    using namespace std::chrono;

    // FILES
    const bool saveData = false;
    std::ifstream load;
    std::ofstream save;
    steady_clock::time_point t1, t2;
    duration<double> dtCenter, dtRev, dtFft, dtT, dtGauss, dtIfft;

	// GRID
	std::string sRows = argv[1];
	const int rows = std::stoi(sRows);
	const int cols = rows;
	const size_t size = rows * cols;
	std::complex<float> *grid = new std::complex<float>[size];
	std::complex<float> *gridT = new std::complex<float>[size];
    const int B = 64;
    int lCols = log2(cols), lRows=log2(rows);
    int revCol[cols], revRow[rows];
    float rN = 1.f / (float) size;
    float rS = 1.f / (2.f * 80.f * 80.f);

    for (int counter=0; counter<1; counter++)
    {
        load.open("data/" + sRows + ".bin", std::ios::binary | std::ios::ate);
//        load.open("data/cats/cut4K2048.bin", std::ios::binary | std::ios::ate);
        std::streamsize nChar = load.tellg();
        load.seekg(0);
        load.read(reinterpret_cast<char *> (grid), nChar);
        load.close();

        t1 = steady_clock::now();
        centerSpectrum(grid, rows, cols);
        t2 = steady_clock::now();
        dtCenter = duration_cast<duration<double>>(t2 - t1);

        // FFT ROWS
        t1 = steady_clock::now();
        #pragma omp parallel for
        for(int j=0; j<cols; j++){
            revCol[j] = revBitOrd(j, lCols);
        }
        t2 = steady_clock::now();
        dtRev = duration_cast<duration<double>>(t2 - t1);

        #pragma omp parallel
        {
            #pragma omp for
            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){
                    gridT[i * cols + revCol[j]] = grid[i*cols + j];
                }
                coolVec(&gridT[i * cols], cols);
            }
            // FFT COLS
            #pragma omp for
            for(int i=0; i<rows; i++){
                revRow[i] = revBitOrd(i, lRows);
            }
        }
        t1 = steady_clock::now();
        transpose(gridT, grid, rows, cols, B);
        t2 = steady_clock::now();
        dtT = duration_cast<duration<double>>(t2 - t1);

        t1 = steady_clock::now();
        #pragma omp parallel for
        for(int i=0; i<cols; i++){
            for(int j=0; j<rows; j++){
                gridT[i*rows + revRow[j]] = grid[i*rows + j];
            }
            coolVec(&gridT[i * rows], rows);
        }
        t2 = steady_clock::now();
        dtFft = duration_cast<duration<double>>(t2 - t1);

        transpose(gridT, grid, cols, rows, B);
        // GAUSSIAN FILTERING
        t1 = steady_clock::now();
        #pragma omp parallel for
        for (int i=0; i<rows; i++){
            int x = i - (rows >> 1);
            for (int j=0; j<cols; j++){
                int y = j - (cols >> 1);
                grid[i*cols + j] *= exp(- (float)(x*x + y*y) * rS);
            }
        }
        t2 = steady_clock::now();
        dtGauss = duration_cast<duration<double>>(t2 - t1);
        // IFFT ROWS
        #pragma omp parallel for
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                gridT[i * cols + revCol[j]] = grid[i*cols + j];
            }
            cevLooc(&gridT[i * cols], cols);
        }
        // IFFT COLS
        transpose(gridT, grid, rows, cols, B);

        t1 = steady_clock::now();
        #pragma omp parallel
        {
            #pragma omp for
            for(int i=0; i<cols; i++){
                for(int j=0; j<rows; j++){
                    gridT[i*rows + revRow[j]] = grid[i*rows + j];
                }
                cevLooc(&gridT[i * rows], rows);
            }

            #pragma omp for
            for (int i=0; i<size; i++){
                gridT[i] *= rN;
            }
        }
        t2 = steady_clock::now();
        dtIfft = duration_cast<duration<double>>(t2 - t1);
        transpose(gridT, grid, cols, rows, B);

        centerSpectrum(grid, rows, cols);
    }

        save.open("logs/singleImage/OMP/centerRevFftTGaussIfftStatic.csv", std::ios::app);
        save << std::getenv("OMP_NUM_THREADS") << ' '
            << sRows << ' '
            << dtCenter.count() << ' '
            << dtRev.count() << ' '
            << dtFft.count() << ' '
            << dtT.count() << ' '
            << dtGauss.count() << ' '
            << dtIfft.count() << std::endl;
        save.close();

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
