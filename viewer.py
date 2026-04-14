#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys


def displayImage(image):
    arr = np.fromfile(image, dtype="float32")
#    arr = np.fromfile(image, dtype="complex64")
#    arr = np.fromfile(image, dtype="uint8")
    arr = arr.reshape(-1, np.sqrt(len(arr)).astype(int)).real
    fig, ax = plt.subplots(figsize=(5,5), layout="constrained")
#    im = ax.imshow(arr, cmap='gray', vmin=0)
    im = ax.imshow(arr, cmap='gray')
    fig.colorbar(im)
    return im


if __name__ == "__main__":
    if len(sys.argv) == 1: raise RuntimeError("Give me something to work with!")
#    size = int(input("What's N? "))
    for img in sys.argv[1:]:
        displayImage(img)
    plt.show()
