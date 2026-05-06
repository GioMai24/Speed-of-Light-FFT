//#include "utilsMP.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>
#include <complex>
#include <numbers>
#include <chrono>

/** @file
 * @brief Benchmark for single ops.
 *
 * Move time_points and change overall op performed. Beware the saving stuff!
 */

int main(int argc, char **argv){
    if (argc != 2){
        std::cout << "Give me some dimensions!" << std::endl;
        return 1;
    }
    using namespace std::chrono;

	// save stuff & time
    std::ifstream load;
    std::ofstream save;
    steady_clock::time_point t1, t2;
    duration<double> dt;


	// grid
	const std::string sRows = argv[1];
	const int rows = std::stoi(sRows);
	const int cols = rows;
	const size_t size = rows * cols;
	std::complex<float> *grid = new std::complex<float>[size];


    t1 = steady_clock::now();
    load.open("data/" + sRows + ".bin", std::ios::binary | std::ios::ate);
	std::streamsize nChar = load.tellg();
	load.seekg(0);
	load.read(reinterpret_cast<char *> (grid), nChar);
	load.close();
    t2 = steady_clock::now();
    dt = duration_cast<duration<double>>(t2 - t1);

    save.open("logs/loadingTime.csv", std::ios::app);
    save << sRows << ' '
        << dt.count() << std::endl;
    save.close();
/*
    t1 = steady_clock::now();
    centerSpectrum(grid, rows, cols);
    t2 = steady_clock::now();
    dtCenter = duration_cast<duration<double>>(t2 - t1);

    save.open("logs/OMP/altDynCenter.csv", std::ios::app);
    save << std::getenv("OMP_NUM_THREADS") << ' '
        << sRows << ' '
        << dtCenter.count() << std::endl;
    save.close();
*/
	delete[] grid;

    return 0;
}
