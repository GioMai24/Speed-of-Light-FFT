<h1 align=center> Speed of Light FFT </h1>

Radix-2 Cooley-Tukey DFT implementation and benchmark of serial, parallel, and GPU parallel code using OpenMP and CUDA.

## Disclaimer

Being my first cpp project I possibly lack basic and well known best practices. I apologise for any heart attack this repo may give you.

<h2 align=center> Project structure </h2>

The heart of the repo resides in the `src/cool` and `src/utils` directories. The former contains the serial, parallel and cuParallel implementations of the DFT. The latter defines the functions used in the implementations. Several functions use templates, and are declared and defined directly in `include/` header files.

```
.
├── include        # headers
│   ├── utils.h                   # serial
│   ├── utilsCu.cuh               # CUDA
│   └── utilsMP.h                 # OpenMP
├── scripts
│   ├── benchmark.sh
│   ├── catter.py
│   ├── checker.py
│   ├── means.py
│   ├── saver.py
│   └── viewer.py
├── src
│   ├── cool
│   │   ├── cool.cc               # serial
│   │   ├── coolCu.cu             # CUDA 
│   │   ├── coolCuDoubleGPU.cu    # CUDA on 2 GPUs
│   │   ├── coolCufft.cu          # cuFFT
│   │   └── coolOmp.cc            # OpenMP
│   ├── tools
│   │   ├── genData.cc
│   │   ├── singleOpBench.cc
│   │   ├── sizeCalc.cc
│   │   └── specs.cu
│   └── utils
│       ├── utils.cc              # serial
│       ├── utilsCu.cu            # CUDA
│       └── utilsMP.cc            # OpenMP
└── visualization.ipynb
```
