#pragma once

void sobel_filter(unsigned char* buffer, float *output, int width, int height, int stride, char type);

void average_pooling(float* buffer, float *output, int width, int height, int stride, int pool_size);
