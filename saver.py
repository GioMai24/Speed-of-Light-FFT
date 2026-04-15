#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from viewer import displayImage

if __name__ == "__main__":
    if len(sys.argv) == 1: raise RuntimeError("Give me something to do!")
    for file in sys.argv[1:]:
        arr = displayImage(file)
        im = Image.fromarray(arr.astype("uint8"))
        im.save(file[:-4] +".jpeg")
    plt.show()
