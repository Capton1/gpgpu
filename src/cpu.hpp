#ifndef CPU_HPP
#define CPU_HPP

#include <string>
#include <stdexcept>
#include <png.h>
#include <cstring>
#include <cassert>
#include <stdio.h>
#include <math.h>
#include "utils.hpp"
#include "image.hpp"

#define PNG_DEBUG 1

Image* sobel(Image *image, const char type);
Image* average_pooling(Image *image, unsigned int pool_size);
Image* post_processing(Image *image, unsigned int postproc_size);
Image* response(Image *image1, Image *image2);
Image* threshold(Image *image);

void process_cpu(const char* img_path);

#endif