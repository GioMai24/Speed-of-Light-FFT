#include "utils.h"
#include <complex>
#include <numbers>


int revBitOrd(int x, int lN){
    int n = 0;
    for(int i=0; i<lN; i++){
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}


void coolVec(std::complex<float> *res, int N){
    for(int s=1; s<=log2(N); s++){
        int m = 1 << s;
        std::complex<float> wm = std::polar(1.f, -2 * std::numbers::pi_v<float> / (float) m);
        for(int k=0; k<N; k+=m){
            std::complex<float> w = 1;
            for(int j=0; j<(m >> 1); j++){
                std::complex<float> t = w * res[k+j+(m >> 1)];
                std::complex<float> u = res[k+j];
                res[k+j] = u + t;
                res[k+j+(m >> 1)] = u - t;
                w *= wm;
            }
        }
    }
}

void coOldVec(std::complex<float> *res, int N){
    for(int s=1; s<=log2(N); s++){
        int m = 1 << s;
        for(int l=0; l<(N >> 1); l++){
            int j = l & ((m >> 1) - 1);  // modulo
            int k = (2 * l >> (int)log2(m)) * m;
            std::complex<float> w = std::polar(1.f, -2 * j * std::numbers::pi_v<float> / (float) m);

            std::complex<float> t = w * res[k + j + (m >> 1)];
            std::complex<float> u = res[k + j];
            res[k + j] = u + t;
            res[k + j + (m >> 1)] = u - t;
        }
    }
}

void cevLooc(std::complex<float> *res, int N){
    for(int s=1; s<=log2(N); s++){
        int m = 1 << s;
        std::complex<float> wm = std::polar(1.f, 2 * std::numbers::pi_v<float> / (float) m);
        for(int k=0; k<N; k+=m){
            std::complex<float> w = 1;
            for(int j=0; j<(m >> 1); j++){
                std::complex<float> t = w * res[k+j+(m >> 1)];
                std::complex<float> u = res[k+j];
                res[k+j] = u + t;
                res[k+j+(m >> 1)] = u - t;
                w *= wm;
            }
        }
    }
}
