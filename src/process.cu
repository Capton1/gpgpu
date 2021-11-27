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

void sobel_filter(const uint8_t* devIn, uint8_t *devOut, int width, int height,
                    int pitchIn, int pitchOut, char type) {

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

__global__ void compute_avg_pooling(const uint8_t* sobelx, const uint8_t* sobely,
                                    uint8_t *out, int patchs_x, int patchs_y,
                                    int pitchX, int pitchY, int pitchOut) {

    __shared__ int local_sum;
    if(threadIdx.x == 0 && threadIdx.y) local_sum = 0;
    __syncthreads();

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < patchs_x * POOLSIZE || y < patchs_y * POOLSIZE)
        atomicAdd(&local_sum, sobelx[(y * pitchX) + x] - sobely[(y * pitchY) + x]);

    __syncthreads();
    if(threadIdx.x == 0 && threadIdx.y) {
        x /= POOLSIZE;
        y /= POOLSIZE;
        float mean = local_sum/(POOLSIZE*POOLSIZE);
        out[x + y * pitchOut] = mean;
    }
}

void average_pooling(const uint8_t* devSobelX, const uint8_t* devSobelY, uint8_t *devOut,
                     int patchs_x, int patchs_y, int pitchX, int pitchY, int pitchOut) {

    int bsize = POOLSIZE;
    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(patchs_x, patchs_y);
    compute_avg_pooling<<<dimGrid, dimBlock>>>(devSobelX, devSobelY, devOut, patchs_x, patchs_y,
                                                pitchX, pitchY, pitchOut);
    cudaDeviceSynchronize();

    if (cudaPeekAtLastError())
        printf("avg pooling Error\n");

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

void threshold(const uint8_t* devIn, uint8_t *devOut, int width, int height,
                int pitchIn, int pitchOut) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned int* devMax;

    rc = cudaMalloc(&devMax, sizeof(unsigned int));
    if (rc)
        printf("Fail max variable allocation\n");

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


void morph_closure(const uint8_t* devIn, uint8_t *devOut,
                    int width, int height, int pitchIn, int pitchOut) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    uint8_t* devTmp;
    size_t pitchTmp;


    rc = cudaMallocPitch(&devTmp, &pitchTmp, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail tmp buffer allocation\n");

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
        
        erosion<<<dimGrid, dimBlock>>>(devTmp, devOut, width, height, pitchTmp, pitchOut);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
            printf("erosion Error\n");
    }

    cudaFree(devTmp);
}


void process_image(const uint8_t* img, uint8_t *output, int width, int height) {
    cudaError_t rc = cudaSuccess;

    int stride_input = width * sizeof(uint8_t);

    // Allocate device memory
    uint8_t *devImg, *devSobelX, *devSobelY;
    size_t pitchImg, pitchX, pitchY;

    rc = cudaMallocPitch(&devImg, &pitchImg, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail devIn allocation\n");

    rc = cudaMemcpy2D(devImg, pitchImg, img, stride_input, width, height, cudaMemcpyHostToDevice);
    if (rc)
        printf("Couldn't copy img to gpu\n");


    // Sobel X
    rc = cudaMallocPitch(&devSobelX, &pitchX, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail devIn allocation\n");
    sobel_filter(devImg, devSobelX, width, height, pitchImg, pitchX, 'x');

    // Sobel Y
    rc = cudaMallocPitch(&devSobelY, &pitchY, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail devIn allocation\n");
    sobel_filter(devImg, devSobelY, width, height, pitchImg, pitchY, 'y');

    // Average Pooling
    int new_width = std::floor((float)width / POOLSIZE);
    int new_height = std::floor((float)height / POOLSIZE);
    int stride_out = new_width * sizeof(uint8_t);

    uint8_t *devResp;
    size_t pitchResp;
    rc = cudaMallocPitch(&devResp, &pitchResp, new_width * sizeof(uint8_t), new_height);
    if (rc)
        printf("Fail devIn allocation\n");
    average_pooling(devSobelX, devSobelY, devResp, new_width, new_height, pitchX, pitchY, pitchResp);

    // Morphological Closure
    uint8_t *devPostproc;
    size_t pitchPostproc;
    rc = cudaMallocPitch(&devPostproc, &pitchPostproc, new_width * sizeof(uint8_t), new_height);
    if (rc)
        printf("Fail devIn allocation\n");
    morph_closure(devResp, devPostproc, new_width, new_height, pitchResp, pitchPostproc);

    // Thresholding
    uint8_t *devOutput;
    size_t pitchOutput;
    rc = cudaMallocPitch(&devOutput, &pitchOutput, new_width * sizeof(uint8_t), new_height);
    if (rc)
        printf("Fail devIn allocation\n");
    threshold(devPostproc, devOutput, new_width, new_height, pitchPostproc, pitchOutput);

    // Copy back to main memory
    rc = cudaMemcpy2D(output, stride_out, devOutput, pitchOutput,
                        new_width * sizeof(uint8_t), new_height, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy output back to memory\n");


    cudaFree(devImg);
    cudaFree(devSobelX);
    cudaFree(devSobelY);
    cudaFree(devResp);
    cudaFree(devPostproc);
    cudaFree(devOutput);
}