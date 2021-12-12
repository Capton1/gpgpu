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


/*
** Scale image by $scale factor.
*/
FIBITMAP* image_scaler(FIBITMAP *image, float scale)
{
    unsigned int width = FreeImage_GetWidth(image);
    unsigned int height = FreeImage_GetHeight(image);

    unsigned int new_width = ceil(width * scale);
    unsigned int new_height = ceil(height * scale);

    FIBITMAP *new_image = FreeImage_Allocate(new_width, new_height, 8);


    for(unsigned int x = 0; x < new_width; x++) {
        for(unsigned int y = 0; y < new_height; y++) {
            uint8_t value;
            FreeImage_GetPixelIndex(image, (unsigned int)floor(x / scale), (unsigned int)floor(y / scale), &value);
            FreeImage_SetPixelIndex(new_image, x, y, &value);
        }
    }

    return new_image;
}
