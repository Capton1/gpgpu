#ifndef UTILS_HPP
#define UTILS_HPP

#include <FreeImage.h>


/*
** Compute Intersection-over-Union metric between ground truth and prediction. 
*/
float compute_IoU(FIBITMAP *gt, FIBITMAP *pred);



#endif