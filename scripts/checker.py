#!/usr/bin/env python
"""Check if original data and FFT->IFFT are the same."""
import numpy as np
import sys 

if __name__=="__main__":
    if len(sys.argv) != 3: raise RuntimeError("I need two files dude!")
    arr1 = np.fromfile(sys.argv[1], dtype="complex64")
    arr2 = np.fromfile(sys.argv[2], dtype="complex64")
    print(np.allclose(arr1, arr2))
    print(np.sum(arr1-arr2))
