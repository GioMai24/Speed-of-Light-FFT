#include <iostream>
#include <fstream>
#include <string>
#include <cuda/std/numbers>
#include <cmath>
#include <cuda/std/complex>
#include <string>


/**
 * The function
 */
float CosCos(const float x, const float y, const float fx, const float fy){
	return cos(2 * cuda::std::numbers::pi * fx * x) * cos(2 * cuda::std::numbers::pi * fy * y);
}


int main(int argc, char **argv){
	/*
	 * Change the following parameters.
	 */ 
	std::string N;
	std::cout << "N? " << std::endl;
	std::cin >> N;
	int rows = std::stoi(N);
	int cols=rows;
	float fx=0.3, fy=0.8, xMax=rows/2, yMax=rows/2;
	std::string name="data/" + N + ".bin";

	std::ofstream save;
	float xTemp, yTemp=0;
	const int size = rows * cols;
	float xStep = xMax / (float) cols;
	float yStep = yMax / (float) rows;
	cuda::std::complex<float> *arr = new cuda::std::complex<float>[size];

	for(int i=0; i<rows; i++){
		xTemp = 0;
		for(int j=0; j<cols; j++){
			arr[i * cols + j] = CosCos(xTemp, yTemp, fx, fy);
			xTemp += xStep;
		}
		yTemp += yStep;
	}

	save.open(name, std::ios::binary);
	save.write(reinterpret_cast<char*> (arr), size * sizeof(cuda::std::complex<float>)/sizeof(char));
	save.close();

	delete[] arr;

	return 0;
}
