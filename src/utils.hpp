#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <png.h>
#include <math.h>
#include "image.hpp"


/*
** Compute Intersection-over-Union metric between ground truth and prediction. 
*/
float compute_IoU(Image *gt, Image *pred);

/*
** Scale image by $scale factor.
*/
Image* image_scaler(Image *image, float scale);

/*
** Read a PNG file.
*/
Image* read_png(const char *filename);

/*
** Write a PNG file.
*/
void write_png(Image* image, const char *filename);


#endif