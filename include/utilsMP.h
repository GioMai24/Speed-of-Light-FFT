#ifndef UTILSMP_H1758
#define UTILSMP_H1758
#include <iostream>
#include <complex>

/** @file
 * @brief parallel functions.
 *
 * Functions for the DFT parallel implementation using OMP. They should appear in this file in roughly the same order they appear in the main code.
 * They are almost identical to utils.h functions, except for the OMP #pragmas.
 */



/**
 * @brief Center image spectrum.
 *
 * Multiplies by -1 every other "bidimensional" array element in a chessboard fashion.
 *
 * @tparam T array datatype.
 * @param[in,out] arr array.
 * @param[in] rows rows of the array.
 * @param[in] cols columns of the array.
 */
template<typename T>
void centerSpectrum(T *arr, const int rows, const int cols){
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
		for(int j=(i & 1); j<cols; j+=2){
			arr[i * cols + j] *= -1;
		}
	}
}


/**
 * @brief Modify index order based on bit representation reversal.
 *
 * Takes in input the index of a vector of length N and returns the bit reversed index.
 *
 * @param x input index.
 * @param lN log2 of vector length.
 * @return n new bit reversed index.
 */
int revBitOrd(int x, int lN);


/**
 * @brief Vector DFT.
 *
 * @param[in,out] res vector.
 * @param[in] N res length.
 */
void coolVec(std::complex<float> *res, int N);

/**
 * @brief Vector IDFT.
 *
 * As coolVec, but inversed.
 *
 * @param[in,out] res vector.
 * @param[in] N res length.
 */
void cevLooc(std::complex<float> *res, int N);


/**
 * @brief Transpose a bidimensional array.
 *
 * Unblocked version with a swap array (two grids, no temp variable).
 *
 * @tparam T array datatype.
 * @param[out] dst transposed array of size cols x rows.
 * @param[in] src input array of size rows x cols.
 * @param[in] rows rows of the array.
 * @param[in] cols columns of the array.
 */
template<typename T>
void transpose(T *src, T *dst, const int rows, const int cols) {
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            dst[j*rows + i] = src[i*cols + j];
        }
    }
}


/**
 * @brief Transpose a bidimensional array.
 *
 * Unblocked version (temp variable).
 *
 * @overload
 * @tparam T array datatype.
 * @param[in,out] src input array of size cols x rows.
 * @param[in] rows rows of the array.
 * @param[in] cols columns of the array.
 */
template<typename T>
void transpose(T *src, const int rows, const int cols) {
    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        for(int j=i+1; j<cols; j++){
            T temp = src[i*cols + j];
            src[i*cols + j] = src[j*rows + i];
            src[j*rows + i] = temp;
        }
    }
}


/**
 * @brief Transpose a bidimensional array.
 *
 * Blocked transpose with square blocks.
 *
 * @overload
 * @tparam T array datatype.
 * @param[out] dst ouput array of size cols x rows.
 * @param[in] src input array of size rows x cols.
 * @param[in] rows rows of the array.
 * @param[in] cols columns of the array.
 * @param[in] B block side.
 */
template<typename T>
void transpose(T *src, T *dst, const int rows, const int cols, const int B) {
    #pragma omp parallel for schedule(dynamic, 1)
    for(int ii=0; ii<rows; ii+=B){
        for(int jj=0; jj<cols; jj+=B){
            for(int i=ii; i<ii+B; i++){
                for(int j=jj; j<jj+B; j++){
                    dst[j*rows + i] = src[i * cols + j];
                }
            }
        }
    }
}

#endif
