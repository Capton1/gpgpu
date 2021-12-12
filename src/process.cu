#include "process.hpp"
#include <stdio.h>

#define TILE_WIDTH 32 + 2 // 32 + r*2

__global__ void sobel_xy(const uint8_t* in, uint8_t *out_x, uint8_t *out_y,
                            int width, int height, int pitchIn,
                            int pitchX, int pitchY) {

    int kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int r = 1;

    __shared__ uint8_t tile[TILE_WIDTH][TILE_WIDTH];

    const int block_x = blockDim.x * blockIdx.x;
    const int block_y = blockDim.y * blockIdx.y;

    int in_x = block_x + threadIdx.x;
    int in_y = block_y + threadIdx.y;

    if (in_x >= width || in_y >= height) return;

    const uint8_t* block_ptr = in + blockIdx.x * blockDim.x
                            + (blockIdx.y * blockDim.y) * pitchIn;
    for (int i = threadIdx.y; i < TILE_WIDTH; i += blockDim.y)
        for (int j = threadIdx.x; j < TILE_WIDTH; j += blockDim.x) {
            int padded_y = i - r;
            int padded_x = j - r;

            // replicate padding
            if (padded_y + block_y < 0)
                padded_y += r;
            if (padded_x + block_x < 0)
                padded_x += r;
            if (padded_y + block_y >= height)
                padded_y -= r;
            if (padded_x + block_x >= width)
                padded_x -= r;

            tile[i][j] = block_ptr[padded_y * pitchIn + padded_x];
        }
    __syncthreads();

    int pix00 = tile[threadIdx.y + 0][threadIdx.x + 0];
    int pix01 = tile[threadIdx.y + 0][threadIdx.x + 1];
    int pix02 = tile[threadIdx.y + 0][threadIdx.x + 2];
    int pix10 = tile[threadIdx.y + 1][threadIdx.x + 0];

    int pix12 = tile[threadIdx.y + 1][threadIdx.x + 2];
    int pix20 = tile[threadIdx.y + 2][threadIdx.x + 0];
    int pix21 = tile[threadIdx.y + 2][threadIdx.x + 1];
    int pix22 = tile[threadIdx.y + 2][threadIdx.x + 2];

    int sumX =  -pix00 + pix02 - 2*pix10 + 2*pix12 - pix20 + pix22;
    int sumY =  pix00 + 2*pix01 + pix02 - pix20 - 2*pix21 - pix22;
    /*for (int kx = 0; kx < 3; kx++) {
        for (int ky = 0; ky < 3; ky++) {
            int pixel = tile[threadIdx.y + ky][threadIdx.x + kx];
            sumX += kernel[ky][kx] * pixel;
            sumY += kernel[kx][2-ky] * pixel;
        }
    }*/

    out_x[in_x + in_y * pitchX] = (sumX > 0) ? sumX : -sumX;
    out_y[in_x + in_y * pitchY] = (sumY > 0) ? sumY : -sumY;
}

void sobel_filter(const uint8_t* devIn, uint8_t *devX, uint8_t *devY,
                    int width, int height, int pitchIn,
                    int pitchX, int pitchY) {

    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    //sobel_x_filter<<<dimGrid, dimBlock>>>(devIn, devX, width, height, pitchIn, pitchX);
    sobel_xy<<<dimGrid, dimBlock>>>(devIn, devX, devY, width, height, pitchIn, pitchX, pitchY);
    cudaDeviceSynchronize();

    if (cudaPeekAtLastError())
        printf("sobel filter Error\n");
}

__global__ void compute_avg_pooling(const uint8_t* sobelx, const uint8_t* sobely,
                                    uint8_t *out, int patchs_x, int patchs_y,
                                    int pitchX, int pitchY, int pitchOut) {

    __shared__ int partialSum[POOLSIZE][POOLSIZE];
    int orig_width = patchs_x * POOLSIZE;
    int orig_height = patchs_y * POOLSIZE;
    int bs = blockDim.x; // blocksize is also equal to blockDim.y

    // Colborative loading
    int tx = threadIdx.x;
    int startx = 2 * blockIdx.x * blockDim.x;
    int x = startx + tx;

    int ty = threadIdx.y;
    int starty = 2 * blockIdx.y * blockDim.y;
    int y = starty + ty;

    partialSum[ty][tx] = 0;
    partialSum[blockDim.y + ty][tx] = 0;
    partialSum[ty][blockDim.x + tx] = 0;
    partialSum[blockDim.y + ty][blockDim.x + tx] = 0;
    if (x < orig_width && y < orig_height)
        partialSum[ty][tx] = sobelx[(y * pitchX) + x] - sobely[(y * pitchY) + x];
    if (x < orig_width && bs + y < orig_height)
        partialSum[bs + ty][tx] = sobelx[((bs + y) * pitchX) + x] - sobely[((bs + y) * pitchY) + x];
    if (bs + x < orig_width && y < orig_height)
        partialSum[ty][bs + tx] = sobelx[(y * pitchX) + x + bs] - sobely[(y * pitchY) + x + bs];
    if (bs + x < orig_width && 16 + y < orig_height)
        partialSum[bs + ty][16 + tx] = sobelx[((bs + y) * pitchX) + x + bs] - sobely[((bs + y) * pitchY) + x + bs];
    __syncthreads();

    // Collaborative reduction
    for(int stride = bs; stride >= 1; stride /= 2) {
        if (ty < stride) {
            partialSum[ty][tx] += partialSum[ty+ stride][tx];
            partialSum[ty][bs + tx] += partialSum[ty+ stride][bs + tx];
        }
        __syncthreads();
    }
    
    for(int stride = bs; stride >= 1; stride /= 2) {
        if (tx < stride) {
            partialSum[ty][tx] += partialSum[ty][tx+ stride];
        }
        __syncthreads();
    }

    // Write to global mem
    if(tx == 0 && ty == 0) {
        float mean = partialSum[0][0]/(POOLSIZE*POOLSIZE);
        out[blockIdx.x + blockIdx.y * pitchOut] = mean;
    }
}

void average_pooling(const uint8_t* devSobelX, const uint8_t* devSobelY, uint8_t *devOut,
                     int patchs_x, int patchs_y, int pitchX, int pitchY, int pitchOut) {

    int bsize = std::ceil((float)POOLSIZE/2);
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


    // Sobel X & Y
    rc = cudaMallocPitch(&devSobelX, &pitchX, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail devSobelX allocation\n");
    rc = cudaMallocPitch(&devSobelY, &pitchY, width * sizeof(uint8_t), height);
    if (rc)
        printf("Fail devSobelY allocation\n");

    sobel_filter(devImg, devSobelX, devSobelY, width, height, pitchImg, pitchX, pitchY);

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
    rc = cudaMemcpy2D(output, width, devSobelY, pitchY,
                        width * sizeof(uint8_t), height, cudaMemcpyDeviceToHost);
    if (rc)
        printf("Unable to copy output back to memory\n");


    cudaFree(devImg);
    cudaFree(devSobelX);
    cudaFree(devSobelY);
    cudaFree(devResp);
    cudaFree(devPostproc);
    cudaFree(devOutput);
}