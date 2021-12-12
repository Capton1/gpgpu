#ifndef UTILS_HPP
#define UTILS_HPP

#include <FreeImage.h>


/*
** Compute Intersection-over-Union metric between ground truth and prediction. 
*/
float compute_IoU(FIBITMAP *gt, FIBITMAP *pred);

/*
** Scale image by $scale factor.
*/
FIBITMAP* image_scaler(FIBITMAP *image, float scale);



#endif