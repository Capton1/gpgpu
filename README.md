# GPGPU: Barcode Detector

```
This project aimed to optimize a simple barcode detector baseline
by re-implementing its image processing functions with CUDA to run on GPU.
```

### Benchmark on the given example image

| Execution time (ms)              | Baseline (CPU) | Our (GPU) |
|----------------------------------|----------------|-----------|
| Sobel filter                     | 3 016          | 0.37      |
| Average pooling                  | 428            | 0.32      |
| Morphological closing            | 29             | 0.009     |
| Thresholding                     | 1              | 0.003     |
| TOTAL (including copying memory) | 5.9s           | 0.45s     |

### Compiling

```sh
mkdir build
cd build
cmake ..
make
```

### Run 

```sh
./main −i image.png −g image−GT.png
```
