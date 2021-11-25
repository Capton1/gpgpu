#pragma once

#include <stdint.h>

void sobel_filter(const uint8_t* buffer, uint8_t *output, int width, int height, int stride, char type);

void average_pooling(const uint8_t* buffer, uint8_t *output, int width, int height, int stride, int pool_size);
