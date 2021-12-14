#include "utils.hpp"
#include <math.h>
#define PNG_DEBUG 1


/*
** Compute Intersection-over-Union metric between ground truth and prediction. 
*/
#include <cstdio>
float compute_IoU(FIBITMAP *gt, FIBITMAP *pred)
{
    float sum_inter = 0;
    float sum_union = 0;

    unsigned int width = FreeImage_GetWidth(gt);
    unsigned int height = FreeImage_GetHeight(gt);

    for(unsigned int x = 0; x < width; x++) {
        for(unsigned int y = 0; y < height; y++) {
            uint8_t value1;
            FreeImage_GetPixelIndex(gt, x, y, &value1);
            uint8_t value2;
            FreeImage_GetPixelIndex(pred, x, y, &value2);

            sum_inter += (value1 & value2) != 0;
            sum_union += (value1 | value2) != 0;
        }
    }
    return sum_inter / sum_union;
}
