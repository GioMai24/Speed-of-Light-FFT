#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys


def displayImage(image):
    arr = np.fromfile(image, dtype="float32").reshape((4096, 4096)).real
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(arr)
    fig.colorbar(im)
    return im


if __name__ == "__main__":
    if len(sys.argv) == 1: raise RuntimeError("Give me something to work with!")
    for img in sys.argv[1:]:
        displayImage(img)
        plt.show()
