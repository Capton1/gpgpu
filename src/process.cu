#include "process.hpp"
#include <stdio.h>

__global__ void compute_filter(const unsigned char* in, unsigned char *out, int width, int height, int r, int sIn, int sOut) {

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < r || x >= width - r) return;
  if (y < r || y >= height - r) return;

  int sum = 0;
  for (int kx = -r; kx <= r; kx++) {
      for (int ky = -r; ky <= r; ky++) {
          sum += in[((y + ky) * sIn) + (x + kx)];
      }
  }
  out[x + y * sOut] = sum / ((2*r+1) * (2*r+1));
}

void sobel_filter(unsigned char* buffer, int width, int height, int stride) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned char*  devIn;
    unsigned char* devOut;
    size_t pitchIn, pitchOut;

    rc = cudaMallocPitch(&devIn, &pitchIn, width * sizeof(char), height);
    if (rc)
        printf("Fail buffer allocation");

    rc = cudaMemcpy2D(devIn, pitchIn, buffer, stride, width, height, cudaMemcpyHostToDevice);
    if (rc)
        printf("Unable to copy buffer back to memory");

    rc = cudaMallocPitch(&devOut, &pitchOut, width * sizeof(char), height);
    if (rc)
        printf("Fail buffer allocation");

    printf("%d, %d, %d, %d\n", width, height, pitchIn, pitchOut);

    {
        int bsize = 32;
        int w     = std::ceil((float)width / bsize);
        int h     = std::ceil((float)height / bsize);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);
        compute_filter<<<dimGrid, dimBlock>>>(devIn, devOut, width, height, 5, pitchIn, pitchOut);
        cudaDeviceSynchronize();

        if (cudaPeekAtLastError())
            printf("compute_filter Error");

    }

    // Copy back to main memory
    rc = cudaMemcpy2D(buffer, stride, devOut, pitchOut, width * sizeof(char), height, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy buffer back to memory");
}