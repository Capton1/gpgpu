#ifndef CPU_HPP
#define CPU_HPP

#include <FreeImage.h>

#define PNG_DEBUG 1

FIBITMAP* sobel(FIBITMAP *image, const char type);
FIBITMAP* average_pooling(FIBITMAP *image, unsigned int pool_size);
FIBITMAP* post_processing(FIBITMAP *image, unsigned int postproc_size);
FIBITMAP* response(FIBITMAP *image1, FIBITMAP *image2);
FIBITMAP* threshold(FIBITMAP *image);

void process_cpu(FIBITMAP *grey);

#endif