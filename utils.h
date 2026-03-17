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

template<typename T>
void spectrum(std::complex<T> *arr, T *res, const int rows, const int cols){
    for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			res[i * cols + j] = abs(arr[i * cols + j]);
		}
	}
}

template<typename T>
void logSpectrum(T *arr, const int rows, const int cols, const T c){
    for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			arr[i * cols + j] = c * log((T)1 + arr[i * cols + j]);
		}
	}
}

#endif
