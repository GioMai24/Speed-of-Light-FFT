#!/usr/bin/env python
import numpy as np
import sys 

if __name__=="__main__":
    if len(sys.argv) != 3: raise RuntimeError("I need two files dude!")
    ser = np.loadtxt(sys.argv[1], delimiter=',')
    omp = np.loadtxt(sys.argv[2], delimiter=',')
    print(np.allclose(ser, omp))
