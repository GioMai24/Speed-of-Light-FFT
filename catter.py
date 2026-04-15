#!/usr/bin/env python
import sys
import numpy as np
from PIL import Image

if __name__ == "__main__":
    if len(sys.argv) == 1: raise RuntimeError("Give me something to do!")
    for file in sys.argv[1:]:
        img = Image.open(file).convert("L")
        np.array(img, dtype="complex64").tofile(file[:-4] + ".bin")
