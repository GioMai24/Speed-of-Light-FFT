#!/usr/bin/env python
import numpy as np
import sys 

if __name__=="__main__":
    if len(sys.argv) != 3: raise RuntimeError("I need two files dude!")
    arr1 = np.fromfile(sys.argv[1], dtype="float32")
    arr2 = np.fromfile(sys.argv[2], dtype="float32")
    print(np.allclose(arr1, arr2))
    print(np.sum(arr1-arr2))
