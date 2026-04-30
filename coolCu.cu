#include <iostream>
#include <fstream>
#include <string>
#include <cuda/cmath>
#include <cuda/std/complex>
#include <cuda/std/numbers>

/**
 * Threads: (half cols, rows) needed.
 */
template<typename T>
__global__ void centerKer(T *arr, const int cols){
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    const int xId = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + (yId & 1);
    arr[yId * cols + xId] *= -1;
}

__global__ void revBitOrdKer(cuda::std::complex<float> *in, cuda::std::complex<float> *out, const int cols){
    int xId = blockDim.x * blockIdx.x + threadIdx.x;
    int n = xId;
    int rXId = 0;
    for(int i=0; i<cuda::ilog2(cols); i++){
        rXId <<= 1;
        rXId |= (n & 1);
        n >>= 1;
    }
    out[blockIdx.y * cols + rXId] = in[blockIdx.y * cols + xId];
}

/**
 * <<<rows, cols>>> limited by max threads x block, and shared memory definition...
 */
__global__ void revBitShOrdKer(cuda::std::complex<float> *in, cuda::std::complex<float> *out, const int cols){
    __shared__ cuda::std::complex<float> help[1024];  // Oh to maually change this... (cols)
    help[threadIdx.x] = in[blockIdx.x * cols + threadIdx.x];
    int xId = threadIdx.x;
    int n = xId;
    int rXId = 0;
    for(int i=0; i<cuda::ilog2(cols); i++){
        rXId <<= 1;
        rXId |= (n & 1);
        n >>= 1;
    }
    __syncthreads();
    out[blockIdx.x * cols + xId] = help[rXId];
}


/**
 * Set <<<(grid.x, rows), threads>>> where grid.x * threads = cols/2
 */
__global__ void coolSubKer(cuda::std::complex<float> *res, const int m, const int cols){
    int xId = blockDim.x * blockIdx.x + threadIdx.x;
    int j = xId & ((m >> 1) - 1);  // modulo
    int k = (2 * xId >> cuda::ilog2(m)) * m;
    cuda::std::complex<float> w = cuda::std::polar(1.f, -2 * j * cuda::std::numbers::pi_v<float> / (float) m);

    cuda::std::complex<float> t = w * res[k + j + (m >> 1) + blockIdx.y * cols];
    cuda::std::complex<float> u = res[k + j + blockIdx.y * cols];
    res[k + j + blockIdx.y * cols] = u + t;
    res[k + j + (m >> 1) + blockIdx.y * cols] = u - t;
}

__global__ void busLoocKer(cuda::std::complex<float> *res, const int m, const int cols){
    int xId = blockDim.x * blockIdx.x + threadIdx.x;
    int j = xId & ((m >> 1) - 1);  // modulo
    int k = (2 * xId >> cuda::ilog2(m)) * m;
    cuda::std::complex<float> w = cuda::std::polar(1.f, 2 * j * cuda::std::numbers::pi_v<float> / (float) m);

    cuda::std::complex<float> t = w * res[k + j + (m >> 1) + blockIdx.y * cols];
    cuda::std::complex<float> u = res[k + j + blockIdx.y * cols];
    res[k + j + blockIdx.y * cols] = u + t;
    res[k + j + (m >> 1) + blockIdx.y * cols] = u - t;
}


template<typename T>
__global__ void gaussKer(T *res, const int cols, const int rows, const float rS){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = xId - (cols >> 1);
    const int y = yId - (rows>> 1);
    res[yId * cols + xId] *= cuda::std::exp(- (float)(x*x + y*y) * rS);
//    res[yId * cols + xId] *= cuda::std::exp(- (float)(x*x + y*y) * 2 * s*s * cuda::std::numbers::pi_v<float>*cuda::std::numbers::pi_v<float>);
}

template<typename T>
__global__ void mulKer(T *res, const int cols, const float rN){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    res[yId * cols + xId] *= rN;
}

template<typename T>
__global__ void subKer(T *arr1, T *arr2, const int cols){
    const int xId = blockDim.x * blockIdx.x + threadIdx.x;
    const int yId = blockDim.y * blockIdx.y + threadIdx.y;
    arr1[yId * cols + xId] -= arr2[yId*cols + xId];
}


template<typename T>
__global__ void sharedTransposeKer(T *in, T *out, const int cols){
    __shared__ T helper[32][33];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    helper[threadIdx.x][threadIdx.y] = in[row * cols + col];
    __syncthreads();
    out[(blockIdx.x * blockDim.x + threadIdx.y) * cols + (blockIdx.y * blockDim.y + threadIdx.x)] = helper[threadIdx.y][threadIdx.x];
}

template<typename T>
__global__ void transposeKer(T *in, T *out, const int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    out[row * cols + col] = in[col * cols + row];
}



int main(int argc, char **argv){
    if (argc != 2){
        std::cout << "Something's off dude..." << std::endl;
        return 1;
    }

	// files
    const bool saveData = false;
    std::ifstream load;
    std::ofstream save;

    // cuda
    cudaStream_t stream, stream2;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&stream2);

//    cudaEvent_t cuT1, cuT2;
//    cudaEventCreate(&cuT1);
//    cudaEventCreate(&cuT2);
//    float cuDt;


	// grid
	const std::string sRows = argv[1];
	const int rows = std::stoi(sRows);
	const int cols = rows;
	const size_t size = rows * cols;
	const size_t cuSize = size * sizeof(cuda::std::complex<float>);

//	uint8_t *img = new uint8_t[size];
    cuda::std::complex<float> *grid = nullptr;
    cuda::std::complex<float> *grid2 = nullptr;
	cuda::std::complex<float> *Dgrid = nullptr;
	cuda::std::complex<float> *Dgrid2 = nullptr;
	cuda::std::complex<float> *DgridT = nullptr;
	cuda::std::complex<float> *DgridT2 = nullptr;

	cudaMallocAsync(&Dgrid, cuSize, stream);
	cudaMallocAsync(&Dgrid2, cuSize, stream2);
	cudaMallocAsync(&DgridT, cuSize, stream);
	cudaMallocAsync(&DgridT2, cuSize, stream2);

	// cuGrid
//    const int blockCols = 16;
//    const int threadsXBlock = cols / 2 / blockCols;
    const int threadsXBlock = cols <= 1024 ? (cols >> 1) : 1024;
    const int blockCols = (cols >> 1) / threadsXBlock;
    dim3 blocks(blockCols, rows);
    dim3 blocksR(blockCols*2, rows);
    dim3 threadsXBlockT(32, 32);
    const int bColsT = cols >> 5, bRowsT = rows >> 5;
    dim3 blocksT(bColsT, bRowsT);
    dim3 blocksC(bColsT >> 1, bRowsT);

	cudaMallocHost(&grid, cuSize, cudaHostAllocDefault);
	cudaMallocHost(&grid2, cuSize, cudaHostAllocDefault);
    for (int counter=0; counter<50; counter++)
    {
        load.open("data/" + sRows + ".bin", std::ios::binary | std::ios::ate);
        std::streamsize nChar = load.tellg();
        load.seekg(0);
        load.read(reinterpret_cast<char *> (grid), nChar);
        load.close();

        load.open("data/" + sRows + "Return.bin", std::ios::binary | std::ios::ate);
        std::streamsize nChar2 = load.tellg();
        load.seekg(0);
        load.read(reinterpret_cast<char *> (grid2), nChar2);
        load.close();

        cudaMemcpyAsync(Dgrid, grid, cuSize, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Dgrid2, grid2, cuSize, cudaMemcpyHostToDevice, stream2);

        centerKer<<<blocksC, threadsXBlockT, 0, stream>>>(Dgrid, cols);
        centerKer<<<blocksC, threadsXBlockT, 0, stream2>>>(Dgrid2, cols);

        // FFT ROWS
    //	cudaEventRecord(cuT1, stream);
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream2>>>(Dgrid2, DgridT2, cols);
    //    revBitShOrdKer<<<1024, 1024, 0, stream>>>(Dgrid, DgridT, cols);
        for(int s=1; s<=log2(cols); s++){
            int m = 1 << s;
            coolSubKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
            coolSubKer<<<blocks, threadsXBlock, 0, stream2>>>(DgridT2, m, cols);
        }

        // FFT COLS (works because square matrix...)
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream2>>>(DgridT2, Dgrid2, cols);
    //   transposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream2>>>(Dgrid2, DgridT2, cols);
    //    revBitShOrdKer<<<1024, 1024, 0, stream>>>(Dgrid, DgridT, cols);
        for(int s=1; s<=log2(cols); s++){
            int m = 1 << s;
            coolSubKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
            coolSubKer<<<blocks, threadsXBlock, 0, stream2>>>(DgridT2, m, cols);
        }
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream2>>>(DgridT2, Dgrid2, cols);
    //    transposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);

        // GAUSSIAN BLUR
        gaussKer<<<blocksT, threadsXBlockT, 0, stream>>>(Dgrid, cols, rows,  1.f / (2.f * 80.f * 80.f));
        gaussKer<<<blocksT, threadsXBlockT, 0, stream2>>>(Dgrid2, cols, rows,  1.f / (2.f * 80.f * 80.f));

        // INVERSE
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream2>>>(Dgrid2, DgridT2, cols);
        for(int s=1; s<=log2(cols); s++){
            int m = 1 << s;
            busLoocKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
            busLoocKer<<<blocks, threadsXBlock, 0, stream2>>>(DgridT2, m, cols);
        }

        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream2>>>(DgridT2, Dgrid2, cols);
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream>>>(Dgrid, DgridT, cols);
        revBitOrdKer<<<blocksR, threadsXBlock, 0, stream2>>>(Dgrid2, DgridT2, cols);
        for(int s=1; s<=log2(cols); s++){
            int m = 1 << s;
            busLoocKer<<<blocks, threadsXBlock, 0, stream>>>(DgridT, m, cols);
            busLoocKer<<<blocks, threadsXBlock, 0, stream2>>>(DgridT2, m, cols);
        }
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream>>>(DgridT, Dgrid, cols);
        sharedTransposeKer<<<blocksT, threadsXBlockT, 0, stream2>>>(DgridT2, Dgrid2, cols);
        mulKer<<<blocksT, threadsXBlockT, 0, stream>>>(Dgrid, cols, 1.f / (float) size);
        mulKer<<<blocksT, threadsXBlockT, 0, stream2>>>(Dgrid2, cols, 1.f / (float) size);

    //    cudaEventRecord(cuT2, stream);
        centerKer<<<blocksC, threadsXBlockT, 0, stream>>>(Dgrid, cols);
        centerKer<<<blocksC, threadsXBlockT, 0, stream2>>>(Dgrid2, cols);
        cudaMemcpyAsync(grid, Dgrid, cuSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(grid2, Dgrid2, cuSize, cudaMemcpyDeviceToHost, stream2);
        cudaStreamSynchronize(stream);
        cudaStreamSynchronize(stream2);
    }
//    centerSpectrum(grid, rows, cols);  // put complex back lol
    // spectrum then log scale
    if (saveData){
        // REAL CASE, NOT FOR COMPLEX ORIGINAL IMAGE!!!!
        float *saveFft = new float[size];
        for(int i=0; i<size; i++){
//            saveFft[i] = log(1.f + hypotf(grid[i].real(), grid[i].imag()));
//            saveFft[i] = hypotf(grid[i].real(), grid[i].imag());
            saveFft[i] = grid[i].real();  // real part use this I guess
        }
//        centerSpectrum(saveFft, rows, cols);
        // COMPLEX CASE
//        std::complex<float> *saveFft = new std::complex<float>[size];
//        for(int i=0; i<size; i++){
//                saveFft[i] = grid[i];
//        }

        save.open("data/fftCu.bin", std::ios::binary);
        save.write(reinterpret_cast<char *> (saveFft), size*sizeof(float) / sizeof(char));
//        save.write(reinterpret_cast<char *> (saveFft), size*sizeof(std::complex<float>) / sizeof(char));
        save.close();
        delete[] saveFft;
    }



//	cudaEventElapsedTime(&cuDt, cuT1, cuT2);
//	std::cout << "Time: " << cuDt << std::endl;

	cudaFreeAsync(Dgrid, stream);
	cudaFreeAsync(Dgrid2, stream2);
	cudaFreeAsync(DgridT, stream);
	cudaFreeAsync(DgridT2, stream2);
	cudaFreeAsync(grid, stream);
	cudaFreeAsync(grid2, stream2);
//    cudaEventDestroy(cuT1);
//    cudaEventDestroy(cuT2);
    cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
    cudaStreamSynchronize(stream2);
	cudaStreamDestroy(stream2);

    return 0;
}
