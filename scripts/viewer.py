#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys


def displayImage(image, dtype="float32"):
    '''
    dtype could be "float32", "complex64", "uint8"
    '''
    arr = np.fromfile(image, dtype=dtype)
    arr = arr.reshape(-1, np.sqrt(len(arr)).astype(int)).real
    fig, ax = plt.subplots(figsize=(5,5), layout="constrained")
    im = ax.imshow(arr, cmap='magma')
    ax.tick_params(labelsize=12)
    fig.colorbar(im)
    return arr


if __name__ == "__main__":
    if len(sys.argv) == 1: raise RuntimeError("Give me something to work with!")
    for img in sys.argv[1:]:
#        displayImage(img, "complex64")
        displayImage(img)
    plt.show()
