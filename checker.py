#!/usr/bin/env python
import numpy as np

if __name__=="__main__":
    ser = np.loadtxt("fftT.csv", delimiter=',')
    omp = np.loadtxt("fft2.csv", delimiter=',')
    print(np.allclose(ser, omp))
