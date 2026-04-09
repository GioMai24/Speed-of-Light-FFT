#include <iostream>
#include <cmath>

int main(){
	int b2Rows;
	int b2Cols;
	int b2Doubt = 6;
	int B = -3;
	int MB = -20;

	std::cout << "Numbers of rows? ";
	std::cin >> b2Rows;
	b2Rows = log2(b2Rows);

	std::cout << "Numbers of cols? ";
	std::cin >> b2Cols;
	b2Cols = log2(b2Cols);

	int size = 1 << (b2Rows + b2Cols + b2Doubt + B + MB);
	std::cout << size << std::endl;
	return 0;
}
