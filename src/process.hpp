#pragma once

#include <stdint.h>
#define POOLSIZE 32

void sobel_filter(const uint8_t* devIn, uint8_t *devX, uint8_t *devY, 
                    int width, int height, int pitchIn,
                    int pitchX, int pitchY);

void average_pooling(const uint8_t* devSobelX, const uint8_t* devSobelY, uint8_t *devOut,
                     int patchs_x, int patchs_y, int pitchX, int pitchY, int pitchOut);

void morph_closure(const uint8_t* devIn, uint8_t *devOut, int width, int height,
                    int pitchIn, int pitchOut);

void threshold(const uint8_t* devIn, uint8_t *devOut, int width, int height,
                int pitchIn, int pitchOut);


void process_image(const uint8_t* img, uint8_t *output, int width, int height);
