#include "process.hpp"
#include <stdio.h>

__global__ void sobel_x_filter(const uint8_t* in, uint8_t *out, int width,
                            int height, int pitchIn, int pitchOut) {

    int kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int r = 1;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < r || x >= width - r) return;
    if (y < r || y >= height - r) return;

    int sum = 0.0;
    for (int kx = -r; kx <= r; kx++) {
        for (int ky = -r; ky <= r; ky++) {
            int pixel = in[((y + ky) * pitchIn) + (x + kx)];
            sum += kernel[ky+r][kx+r] * pixel;
        }
    }

    out[x + y * pitchOut] = (sum > 0) ? sum : -sum;
}

__global__ void sobel_y_filter(const uint8_t* in, uint8_t *out, int width,
                            int height, int pitchIn, int pitchOut) {

    int kernel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    int r = 1;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < r || x >= width - r) return;
    if (y < r || y >= height - r) return;

    int sum = 0;
    for (int kx = -r; kx <= r; kx++) {
        for (int ky = -r; ky <= r; ky++) {
            int pixel = in[((y + ky) * pitchIn) + (x + kx)];
            sum += kernel[ky+r][kx+r] * pixel;
        }
    }

    out[x + y * pitchOut] = (sum > 0) ? sum : -sum;
}

void sobel_filter(const uint8_t* buffer, uint8_t *output, int width, int height, int stride, char type) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    uint8_t* devIn;
    uint8_t* devOut;
    size_t pitchIn, pitchOut;

    rc = cudaMallocPitch(&devIn, &pitchIn, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail buffer allocation\n");

    rc = cudaMemcpy2D(devIn, pitchIn, buffer, stride, width, height, cudaMemcpyHostToDevice);
    if (rc)
        printf("Couldn't copy data to gpu\n");

    rc = cudaMallocPitch(&devOut, &pitchOut, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail buffer allocation\n");

    {
        int bsize = 32;
        int w     = std::ceil((float)width / bsize);
        int h     = std::ceil((float)height / bsize);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);
        if (type == 'x')
            sobel_x_filter<<<dimGrid, dimBlock>>>(devIn, devOut, width, height, pitchIn, pitchOut);
        else
            sobel_y_filter<<<dimGrid, dimBlock>>>(devIn, devOut, width, height, pitchIn, pitchOut);
        cudaDeviceSynchronize();

        if (cudaPeekAtLastError())
            printf("sobel filter Error\n");

    }

    // Copy back to main memory
    rc = cudaMemcpy2D(output, stride * sizeof(uint8_t), devOut, pitchOut,
                        width * sizeof(uint8_t), height, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy buffer back to memory\n");

    cudaFree(devIn);
    cudaFree(devOut);
}


__global__ void compute_avg_pooling(const uint8_t* sobelx, const uint8_t* sobely,
                                    uint8_t *out, int patchs_x, int patchs_y,
                                    int pool_size, int pitchIn, int pitchOut) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= patchs_x || y >= patchs_y) return;

    float sumX = 0;
    float sumY = 0;
    int index_y = y * pool_size;
    int index_x = x * pool_size;
    for (int i = index_x; i <= index_x + pool_size; i++) {
        for (int j = index_y; j <= index_y + pool_size; j++) {
            sumX += sobelx[(j * pitchIn) + i];
            sumY += sobely[(j * pitchIn) + i];
        }
    }
    sumX /= (pool_size*pool_size);
    sumY /= (pool_size*pool_size);
    out[x + y * pitchOut] = (sumX - sumY);
}


void average_pooling(const uint8_t* sobel_x, const uint8_t* sobel_y, uint8_t *output,
                     int width, int height, int stride, int pool_size) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    uint8_t* devSobelX;
    uint8_t* devSobelY;
    uint8_t* devOut;
    size_t pitchIn, pitchOut;
    int patchs_x = std::floor((float)width / pool_size);
    int patchs_y = std::floor((float)height / pool_size);

    rc = cudaMallocPitch(&devSobelX, &pitchIn, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail buffer allocation\n");
    rc = cudaMallocPitch(&devSobelY, &pitchIn, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail buffer allocation\n");

    rc = cudaMemcpy2D(devSobelX, pitchIn, sobel_x, stride, width, height, cudaMemcpyHostToDevice);
    if (rc)
        printf("Couldn't copy data to gpu\n");
    rc = cudaMemcpy2D(devSobelY, pitchIn, sobel_y, stride, width, height, cudaMemcpyHostToDevice);
    if (rc)
        printf("Couldn't copy data to gpu\n");

    rc = cudaMallocPitch(&devOut, &pitchOut, patchs_x * sizeof(uint8_t), patchs_y);
    if (rc)
        printf("Fail buffer allocation\n");

    {
        dim3 dimBlock(1, 1);
        dim3 dimGrid(patchs_x, patchs_y);
        compute_avg_pooling<<<dimGrid, dimBlock>>>(devSobelX, devSobelY, devOut, patchs_x,
                                                    patchs_y, pool_size, pitchIn, pitchOut);
        cudaDeviceSynchronize();

        if (cudaPeekAtLastError())
            printf("avg pooling Error\n");

    }

    // Copy back to main memory
    rc = cudaMemcpy2D(output, patchs_x * sizeof(uint8_t), devOut, pitchOut,
                        patchs_x * sizeof(uint8_t), patchs_y, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy buffer back to memory\n");

    cudaFree(devSobelX);
    cudaFree(devSobelY);
    cudaFree(devOut);
}


__global__ void compute_max(const uint8_t* in, unsigned int *max,
                            int width, int height, int pitchIn) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    atomicMax(max, in[x + y * pitchIn]);
}

__global__ void compute_threshold(const uint8_t* in,
                                    uint8_t *out, unsigned int *max, int width, int height,
                                    int pitchIn, int pitchOut) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;
    uint8_t value = *max/2;
    out[x + y * pitchOut] = 255 * (in[x + y * pitchIn] > value);
}

void threshold(const uint8_t* buffer, uint8_t *output,
                int width, int height, int stride) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    uint8_t* devIn;
    unsigned int* devMax;
    uint8_t* devOut;
    size_t pitchIn, pitchOut;

    rc = cudaMallocPitch(&devIn, &pitchIn, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail buffer allocation\n");

    rc = cudaMalloc(&devMax, sizeof(unsigned int));
    if (rc)
        printf("Fail max allocation\n");

    rc = cudaMemcpy2D(devIn, pitchIn, buffer, stride, width, height, cudaMemcpyHostToDevice);
    if (rc)
        printf("Couldn't copy data to gpu\n");

    rc = cudaMallocPitch(&devOut, &pitchOut, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail buffer allocation\n");

    {
        int bsize = 32;
        int w     = std::ceil((float)width / bsize);
        int h     = std::ceil((float)height / bsize);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);

        compute_max<<<dimGrid, dimBlock>>>(devIn, devMax, width, height, pitchIn);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
            printf("max Error\n");
        
        compute_threshold<<<dimGrid, dimBlock>>>(devIn, devOut, devMax, width, height, pitchIn, pitchOut);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
            printf("thresholding Error\n");

    }

    // Copy back to main memory
    rc = cudaMemcpy2D(output, stride * sizeof(uint8_t), devOut, pitchOut,
                        width * sizeof(uint8_t), height, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy buffer back to memory\n");

    cudaFree(devIn);
    cudaFree(devOut);
    cudaFree(devMax);
}


__global__ void dilation(const uint8_t* in, uint8_t *out, int width,
                            int height, int pitchIn, int pitchOut) {

    int kernel[5][5] = {{0, 0, 0, 0, 0}, {255, 255, 255, 255, 255}, {255, 255, 255, 255, 255},
                        {255, 255, 255, 255, 255}, {0, 0, 0, 0, 0}};
    int r = 2;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t current_val = in[x + y * pitchIn];
    for (int kx = -r; kx <= r; kx++) {
        for (int ky = -r; ky <= r; ky++) {
            if (kernel[ky+r][kx+r] == 0) continue;
            if (y + ky < 0 && y + ky >= height) continue;
            if (x + kx < 0 && x + kx >= width) continue;
            int prop_val = in[((y + ky) * pitchIn) + (x + kx)];
            if (prop_val > current_val)
                current_val = prop_val;
        }
    }

    out[x + y * pitchOut] = current_val;
}

__global__ void erosion(const uint8_t* in, uint8_t *out, int width,
                            int height, int pitchIn, int pitchOut) {

    int kernel[5][5] = {{0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0}};
    int r = 2;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t current_val = in[x + y * pitchIn];
    for (int kx = -r; kx <= r; kx++) {
        for (int ky = -r; ky <= r; ky++) {
            if (kernel[ky+r][kx+r] == 0) continue;
            if (y + ky < 0 && y + ky >= height) continue;
            if (x + kx < 0 && x + kx >= width) continue;
            int prop_val = in[((y + ky) * pitchIn) + (x + kx)];
            if (prop_val < current_val)
                current_val = prop_val;
        }
    }

    out[x + y * pitchOut] = current_val;
}


void morph_closure(const uint8_t* buffer, uint8_t *output,
                int width, int height, int stride) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    uint8_t* devIn;
    uint8_t* devTmp;
    uint8_t* devOut;
    size_t pitchIn, pitchTmp, pitchOut;

    rc = cudaMallocPitch(&devIn, &pitchIn, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail buffer allocation\n");

    rc = cudaMemcpy2D(devIn, pitchIn, buffer, stride, width, height, cudaMemcpyHostToDevice);
    if (rc)
        printf("Couldn't copy data to gpu\n");

    rc = cudaMallocPitch(&devTmp, &pitchTmp, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail buffer allocation\n");

    rc = cudaMallocPitch(&devOut, &pitchOut, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail buffer allocation\n");

    {
        int bsize = 32;
        int w     = std::ceil((float)width / bsize);
        int h     = std::ceil((float)height / bsize);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);

        dilation<<<dimGrid, dimBlock>>>(devIn, devTmp, width, height, pitchIn, pitchTmp);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
            printf("dilation Error\n");
        
        erosion<<<dimGrid, dimBlock>>>(devTmp, devOut, width, height, pitchIn, pitchOut);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
            printf("erosion Error\n");

    }

    // Copy back to main memory
    rc = cudaMemcpy2D(output, stride * sizeof(uint8_t), devOut, pitchOut,
                        width * sizeof(uint8_t), height, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy buffer back to memory\n");

    cudaFree(devIn);
    cudaFree(devTmp);
    cudaFree(devOut);
}