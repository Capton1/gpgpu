#include "process.hpp"
#include <stdio.h>

__global__ void sobel_x_filter(const unsigned char* in, float *out, int width,
                            int height, int pitch) {

    float kernel[3][3] = {{-1.0, 0.0, 1.0}, {-2.0, 0.0, 2.0}, {-1.0, 0.0, 1.0}};
    int r = 1;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < r || x >= width - r) return;
    if (y < r || y >= height - r) return;

    float sum = 0.0;
    for (int kx = -r; kx <= r; kx++) {
        for (int ky = -r; ky <= r; ky++) {
            float pixel = in[((y + ky) * pitch) + (x + kx)];
            sum += kernel[ky+r][kx+r] * pixel;
        }
    }

    out[x + y * pitch] = (sum > 0) ? sum : -sum;
}

__global__ void sobel_y_filter(const unsigned char* in, float *out, int width,
                            int height, int pitch) {

    float kernel[3][3] = {{1.0, 2.0, 1.0}, {0.0, 0.0, 0.0}, {-1.0, -2.0, -1.0}};
    int r = 1;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < r || x >= width - r) return;
    if (y < r || y >= height - r) return;

    float sum = 0.0;
    for (int kx = -r; kx <= r; kx++) {
        for (int ky = -r; ky <= r; ky++) {
            float pixel = in[((y + ky) * pitch) + (x + kx)];
            sum += kernel[ky+r][kx+r] * pixel;
        }
    }

    out[x + y * pitch] = (sum > 0) ? sum : -sum;
}

void sobel_filter(unsigned char* buffer, float *filter_output, int width, int height, int stride, char type) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned char*  devIn;
    float* devOut;
    size_t pitchIn, pitchOut;

    rc = cudaMallocPitch(&devIn, &pitchIn, width * sizeof(char), height);
    if (rc)
        printf("Fail buffer allocation\n");

    rc = cudaMemcpy2D(devIn, pitchIn, buffer, stride, width, height, cudaMemcpyHostToDevice);
    if (rc)
        printf("Unable to copy buffer back to memory\n");

    rc = cudaMallocPitch(&devOut, &pitchOut, width * sizeof(float), height);
    if (rc)
        printf("Fail buffer allocation\n");

    {
        int bsize = 32;
        int w     = std::ceil((float)width / bsize);
        int h     = std::ceil((float)height / bsize);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);
        if (type == 'x')
            sobel_x_filter<<<dimGrid, dimBlock>>>(devIn, devOut, width, height, pitchIn);
        else
            sobel_y_filter<<<dimGrid, dimBlock>>>(devIn, devOut, width, height, pitchIn);
        cudaDeviceSynchronize();

        if (cudaPeekAtLastError())
            printf("compute_filter Error\n");

    }

    // Copy back to main memory
    rc = cudaMemcpy2D(filter_output, stride * sizeof(float), devOut, pitchOut,
                        width * sizeof(float), height, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy buffer back to memory\n");
}