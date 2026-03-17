#include <iostream>
#include <fstream>  // file handler
#include <ctime>
#include <cmath>
#include <cstring>  // memset
#include <complex>

#ifndef PI
#define PI 3.14159265
#endif
const std::complex<float> i(0,1);

float CosCos(const float x, const float y, const float fx, const float fy){
	return cos(2 * PI * fx * x) * cos(2 * PI * fy * y);
}


template <typename T>
void printArray(T *arr, int rows, int cols){
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			std::cout << arr[i * cols + j] << ' ';
		}
		std::cout << std::endl;
	}
}


int main(){
	// frequencies
	const float fx = 0.5;
	const float fy = 1;

	// points
	float xMin = -100, xMax = 100;
	float yMin = -100, yMax = 100;
	// would be nice to throw an error if min > max

	// grid
	const int rows = yMax - yMin;
	const int cols = xMax - xMin;
	float grid[rows * cols];
	memset(grid, 0, sizeof(grid));  // initialize to 0, unnecessary
	float xStep = (xMax - xMin) / (float) cols;
	float yStep = (yMax - yMin) / (float) rows;

	float xTemp;
	for(int i=0; i<rows; i++){
		xTemp = xMin;
		for(int j=0; j<cols; j++){
			grid[i * cols + j] = CosCos(xTemp, yMin, fx, fy);
			xTemp += xStep;
		}
		yMin += yStep;
	}

	std::ofstream aFile;
	aFile.open("data.csv");
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			aFile << grid[i * cols + j];
			if(j != cols - 1){aFile << ", ";}
		}
		aFile << '\n';
	}
	aFile.close();
	return 0;
}
