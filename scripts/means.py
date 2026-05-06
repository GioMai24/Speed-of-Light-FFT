#!/usr/bin/env python
import sys
import numpy as np
from pathlib import Path

if __name__=="__main__":
    if len(sys.argv) != 2: raise RuntimeError("Nuh-hu")
    times = {entry: ((arr := np.loadtxt(entry)).mean(), arr.std()) for entry in Path(sys.argv[1]).glob("**/*.txt")}
    for path, val in sorted(times.items(), key=lambda item: item[1][0]):
        print(path)
        print(val[0], val[1], '\n')
