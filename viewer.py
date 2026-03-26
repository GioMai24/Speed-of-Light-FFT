#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np


def displayImage(image):
    arr = np.loadtxt(image, delimiter=',')
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(arr)
    fig.colorbar(im)
    return im


if __name__ == "__main__":
#    displayImage("data.csv")
#    plt.show()

    displayImage("fftAngle.csv")
    plt.show()
