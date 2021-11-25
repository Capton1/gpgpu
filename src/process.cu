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

void sobel_filter(unsigned char* buffer, float *output, int width, int height, int stride, char type) {
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
            printf("sobel filter Error\n");

    }

    // Copy back to main memory
    rc = cudaMemcpy2D(output, stride * sizeof(float), devOut, pitchOut,
                        width * sizeof(float), height, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy buffer back to memory\n");
}


__global__ void compute_avg_pooling(const float* in, float *out, int patchs_x,
                                    int patchs_y, int pool_size, int pitchIn, int pitchOut) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= patchs_x || y >= patchs_y) return;

    float sum = 0.0;
    int index_y = y * pool_size;
    int index_x = x * pool_size;
    for (int i = index_x; i <= index_x + pool_size; i++) {
        for (int j = index_y; j <= index_y + pool_size; j++) {
            sum += in[(j * pitchIn/sizeof(float)) + i];
        }
    }

    out[x + y * patchs_x] = 255.0;//sum / (pool_size*pool_size);
}


void average_pooling(float* buffer, float *output, int width, int height, int stride, int pool_size) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    float*  devIn;
    float* devOut;
    size_t pitchIn, pitchOut;
    int patchs_x = std::floor((float)width / pool_size);
    int patchs_y = std::floor((float)height / pool_size);

    printf("%d %d\n", patchs_x, patchs_y);

    rc = cudaMallocPitch(&devIn, &pitchIn, width * sizeof(float), height);
    if (rc)
        printf("Fail buffer allocation\n");

    rc = cudaMemcpy2D(devIn, pitchIn, buffer, stride, width, height, cudaMemcpyHostToDevice);
    if (rc)
        printf("Unable to copy buffer back to memory\n");

    rc = cudaMallocPitch(&devOut, &pitchOut, patchs_x * sizeof(float), patchs_y);
    if (rc)
        printf("Fail buffer allocation\n");

    {
        dim3 dimBlock(1, 1);
        dim3 dimGrid(patchs_x, patchs_y);
        compute_avg_pooling<<<dimGrid, dimBlock>>>(devIn, devOut, patchs_x, patchs_y, pool_size, pitchIn, pitchOut);
        cudaDeviceSynchronize();

        if (cudaPeekAtLastError())
            printf("avg pooling Error\n");

    }

    // Copy back to main memory
    rc = cudaMemcpy2D(output, stride, devOut, pitchOut,
                        patchs_x * sizeof(float), patchs_y, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy buffer back to memory\n");
}