#include "utils.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <complex>

/** @file
 * @brief Serial DFT implementation.
 *
 * Compute 100 "images".
 */



int main(int argc, char **argv){
    if (argc != 2){
        std::cout << "Give me some dimensions!" << std::endl;
        return 1;
    }

    // FILES
    const bool saveData = false;
    std::ifstream load;
    std::ofstream save;

	// GRID
	const std::string sRows = argv[1];
	const int rows = std::stoi(sRows);
	const int cols = rows;
	const size_t size = rows * cols;
	std::complex<float> *grid = new std::complex<float>[size];
	std::complex<float> *gridT = new std::complex<float>[size];
    const int B = 8;
    const int lCols = log2(cols), lRows = log2(rows);
    int revCol[cols], revRow[rows];
    const float rN = 1.f / (float) size;
    const float rS = 1.f / (2.f * 80.f * 80.f);

    for (int counter=0; counter<100; counter++)
    {
        load.open("data/" + sRows + ".bin", std::ios::binary | std::ios::ate);
        std::streamsize nChar = load.tellg();
        load.seekg(0);
        load.read(reinterpret_cast<char *> (grid), nChar);
        load.close();

        centerSpectrum(grid, rows, cols);

        // FFT ROWS
        for(int j=0; j<cols; j++){
            revCol[j] = revBitOrd(j, lCols);
        }

        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                gridT[i * cols + revCol[j]] = grid[i*cols + j];
            }
            coolVec(&gridT[i * cols], cols);
        }

        // FFT COLS
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

        // GAUSSIAN FILTERING
        for (int i=0; i<rows; i++){
            int x = i - (rows >> 1);
            for (int j=0; j<cols; j++){
                int y = j - (cols >> 1);
                grid[i*cols + j] *= exp(- (float)(x*x + y*y) * rS);
            }
        }

        // IFFT ROWS
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                gridT[i * cols + revCol[j]] = grid[i*cols + j];
            }
            cevLooc(&gridT[i * cols], cols);
        }

        // IFFT COLS
        transpose(grid, gridT, rows, cols, B);
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                gridT[i*cols + revRow[j]] = grid[i*cols + j];
            }
            cevLooc(&gridT[i * cols], cols);
        }
        for (int i=0; i<size; i++){
            gridT[i] *= rN;
        }
        transpose(grid, gridT, rows, cols, B);

        centerSpectrum(grid, rows, cols);
    }

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
	delete[] gridT;

    return 0;
}
