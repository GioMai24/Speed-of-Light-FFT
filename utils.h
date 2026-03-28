#ifndef UTILS_H
#define UTILS_H

template<typename T>
void printArray(T *arr, const int rows, const int cols){
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			std::cout << arr[i * cols + j] << ' ';
		}
		std::cout << std::endl;
	}
}

template<typename T>
void transpose(T *src, T *dst, const int rows, const int cols) {
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            dst[j*rows + i] = src[i * cols + j];
        }
    }
}


template<typename T>
void transpose(T *src, T *dst, const int rows, const int cols, const int B) {
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


/**
 * Modify index order based on bit representation reversal.
 *
 * Example: x=3 -> 011 -> 110 -> 6
 */
int revBitOrd(int x, int lN){
    int n = 0;
    for(int i=0; i<lN; i++){
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}


/**
 * To be applied to the image before DFT.
 */
template<typename T>
void centerSpectrum(T *arr, const int rows, const int cols){
    for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			arr[i * cols + j] *= pow(-1, i+j);
		}
	}
}

#endif
