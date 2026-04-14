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


/**
 * Unblocked transpose. swap version (two grids, no temp)
 */
template<typename T>
void transpose(T *dst, T *src, const int rows, const int cols) {
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            dst[i*cols + j] = src[j*rows + i];
        }
    }
}


/**
 * Unblocked transpose. Temp variable version
 */
template<typename T>
void transpose(T *src, const int rows, const int cols) {
    for(int i=0; i<rows; i++){
        for(int j=i+1; j<cols; j++){
            T temp = src[i*cols + j];
            src[i*cols + j] = src[j*rows + i];
            src[j*rows + i] = temp;
        }
    }
}


/**
 * Blocked transpose. two grids necessary + all cols comp.
 */
template<typename T>
void transpose(T *dst, T *src, const int rows, const int cols, const int B) {
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

int *read_pgm(const char *filename, int *width, int *height, int *max_val) {

    // Open the input file in read mode "r"
    FILE *file = fopen(filename, "r");

    // Check if file can be opened
    if (file == NULL) {
        printf("Could not open file.\n");
        return NULL;
    }

    // Read the PGM header, composed by 3 lines, e.g.:
    //
    // P2                           [magic number]
    // 1024 768                     [pixel_width pixel_height]
    // 255                          [max grayscale levels]
    //
    // More info here -- https://www.wikiwand.com/en/articles/Netpbm

    // Read the first line, and verify if it states `P2`
    char format[3];
    fscanf(file, "%s", format);
    if (format[0] != 'P' || format[1] != '2') {
        printf("Not a valid PGM (ASCII P2) file.\n");
        fclose(file);
        return NULL;
    }

    // Read the width, height, and maximum grayscale value
    fscanf(file, "%d %d", width, height);
    fscanf(file, "%d", max_val);

    // Compute the total amount of pixels
    int total_pixels = (*width) * (*height);

    // Allocate host memory for the image data
    int *image = (int *)malloc(total_pixels * sizeof(int));

    // Read pixel values into the array
    for (int i = 0; i < total_pixels; i++) {
        fscanf(file, "%d", &image[i]);
    }

    // Close the input file
    fclose(file);

    // Return the pixel array
    return image;
}


#endif
