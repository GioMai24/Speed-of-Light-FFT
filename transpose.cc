#include <iostream>
#include <fstream>
#include <chrono>
#include <complex>
#include "utils.h"

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
    int B = atoi(argv[1]);
    load.open("data/8192.bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (grid), nChar);
	load.close();

	t1 = steady_clock::now();
	transpose(grid, gridT, rows, cols, B);
	t2 = steady_clock::now();
	dt = duration_cast<duration<double>>(t2 - t1);
	std::cout << dt.count() << std::endl;

	return 0;
}
