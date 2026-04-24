#include <iostream>
#include <fstream>
#include <chrono>
#include <complex>
#include "utilsMP.h"

int main(int argc, char **argv){
    if (argc==1){
        std::cout << "Missing block size" << std::endl;
        return 1;
    }

    using namespace std::chrono;
	std::ifstream load;
	steady_clock::time_point t1, t2;
    duration<double> dt;

	const int rows = 8192;
	const int cols = rows;
	const size_t size = rows * cols;
	std::complex<float> *grid = new std::complex<float>[size];
	std::complex<float> *gridT = new std::complex<float>[size];
    int Boi = atoi(argv[1]);
    int B = 64;

    load.open("data/8192.bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (grid), nChar);
	load.close();


	t1 = steady_clock::now();
	#pragma omp parallel
	{
 		#pragma omp for schedule(dynamic, Boi)
	    for(int ii=0; ii<rows; ii+=B){
	        for(int jj=0; jj<cols; jj+=B){
	            for(int i=ii; i<ii+B; i++){
	                for(int j=jj; j<jj+B; j++){
	                    gridT[j*rows + i] = grid[i * cols + j];
	                }
	            }
    	    }
	    }	
	}
	t2 = steady_clock::now();
	dt = duration_cast<duration<double>>(t2 - t1);
	std::cout << dt.count() << std::endl;
	delete[] grid;
	delete[] gridT;
	return 0;
}
