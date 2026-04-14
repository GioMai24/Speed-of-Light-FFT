#include <cstdint>
#include <iostream>
#include <cmath>
#include <fstream>
#include "utils.h"

int main(){
	int width, height, max_val;
	int *img = read_pgm("data/cats/pazz.pgm", &width, &height, &max_val);
	uint8_t *newImg = new uint8_t[512*512];
	std::ofstream save;

	for (int i=0; i<512; i++){
		for (int j=0; j<512; j++){
			newImg[i * 512+ j] = (uint8_t) img[i*width + j];
		}
	}

	save.open("data/cats/cat512.bin", std::ios::binary);
	save.write(reinterpret_cast<char *> (newImg), 512*512*sizeof(uint8_t)/sizeof(char));
	save.close();

	free(img);
	delete[] newImg;
	return 0;
}
