#include <iostream>
#include <fstream>
#include <string>
#include <numbers>
#include <cmath>
#include <complex>

/** @file
 * @brief Generate square data to transform.
 */



/**
 * @brief Testing cosine function
 *
 * @param x x coordinate.
 * @param y y coordinate.
 * @param fx x frequency.
 * @param fy y frequency.
 * @return CosCos value.
 */
float CosCos(const float x, const float y, const float fx, const float fy){
	return cos(2 * std::numbers::pi * fx * x) * cos(2 * std::numbers::pi * fy * y);
}


int main(int argc, char **argv){
	/*
	 * Change the following parameters.
	 */
	std::string N;  // side length
	std::cout << "N? " << std::endl;
	std::cin >> N;
	int rows = std::stoi(N);
	int cols = rows;
	float fx = 0.15, fy = 0.4;
	std::string name="data/" + N + ".bin";

	std::ofstream save;
	const size_t size = rows * cols;
	std::complex<float> *arr = new std::complex<float>[size];

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			arr[i * cols + j] = CosCos(j, i, fx, fy);
		}
	}

	save.open(name, std::ios::binary);
	save.write(reinterpret_cast<char*> (arr), size * sizeof(std::complex<float>)/sizeof(char));
	save.close();

	delete[] arr;

	return 0;
}
