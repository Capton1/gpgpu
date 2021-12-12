#include "cpu.hpp"
#include <algorithm>

#define DEBUG true

FIBITMAP* sobel(FIBITMAP *image, const char type) {

    unsigned int width = FreeImage_GetWidth(image);
    unsigned int height = FreeImage_GetHeight(image);
    FIBITMAP *new_image = FreeImage_Allocate(width, height, 8);

    if (type == 'x') {

        for(unsigned int y = 1; y < height - 1; y++) {
            for(unsigned int x = 1; x < width - 1; x++) {
                int sum = 0;
                uint8_t value;

                FreeImage_GetPixelIndex(image, x - 1, y - 1, &value);
                sum += value * -1;
                FreeImage_GetPixelIndex(image, x + 1, y - 1, &value);
                sum += value * 1;

                FreeImage_GetPixelIndex(image, x - 1, y, &value);
                sum += value * -2;
                FreeImage_GetPixelIndex(image, x + 1, y, &value);
                sum += value * 2;

                FreeImage_GetPixelIndex(image, x - 1, y + 1, &value);
                sum += value * -1;
                FreeImage_GetPixelIndex(image, x + 1, y + 1, &value);
                sum += value * 1;

                sum = std::abs(sum);
                sum = std::max(0, std::min(sum, 255));

                BYTE val = sum;
                FreeImage_SetPixelIndex(new_image, x, y, &val);
            }

        }
    }
    else if (type == 'y') {
        for(unsigned int y = 1; y < height - 1; y++) {
            for(unsigned int x = 1; x < width - 1; x++) {

                int sum = 0;
                uint8_t value;

                FreeImage_GetPixelIndex(image, x - 1, y - 1, &value);
                sum += value * -1;
                FreeImage_GetPixelIndex(image, x, y - 1, &value);
                sum += value * -2;
                FreeImage_GetPixelIndex(image, x + 1, y - 1, &value);
                sum += value * -1;


                FreeImage_GetPixelIndex(image, x - 1, y + 1, &value);
                sum += value * 1;
                FreeImage_GetPixelIndex(image, x, y + 1, &value);
                sum += value * 2;
                FreeImage_GetPixelIndex(image, x + 1, y + 1, &value);
                sum += value * 1;

                sum = std::abs(sum);
                sum = std::max(0, std::min(sum, 255));

                BYTE val = sum;
                FreeImage_SetPixelIndex(new_image, x, y, &val);
            }
        }
    }

    return new_image;
}

FIBITMAP* average_pooling(FIBITMAP *image, unsigned int pool_size) {
    unsigned int width = FreeImage_GetWidth(image);
    unsigned int height = FreeImage_GetHeight(image);

    unsigned int new_width = width / pool_size;
    unsigned int new_height = height / pool_size;
    FIBITMAP *new_image = FreeImage_Allocate(new_width, new_height, 8);

    for (unsigned int ii = 0; ii < new_height; ii++) {
        for (unsigned int jj = 0; jj < new_width; jj++) {
            unsigned int sum = 0;
            int index_y = ii * pool_size;
            int index_x = jj * pool_size;
            for (unsigned int i = index_y; i < index_y + pool_size; i++) {
                for (unsigned int j = index_x; j < index_x + pool_size; j++) {
                    uint8_t value;
                    FreeImage_GetPixelIndex(image, j, i, &value);
                    sum += value;
                }
            }
            BYTE val = sum / (pool_size * pool_size);
            FreeImage_SetPixelIndex(new_image, jj, ii, &val);
        }
    }
    
    return new_image;
}


FIBITMAP* post_processing(FIBITMAP *image, unsigned int postproc_size) {

    unsigned int width = FreeImage_GetWidth(image);
    unsigned int height = FreeImage_GetHeight(image);

    FIBITMAP *new_image = FreeImage_Allocate(width, height, 8);
    FIBITMAP *buffer = FreeImage_Allocate(width, height, 8);
    FIBITMAP *tmp = FreeImage_Allocate(postproc_size, postproc_size, 8);

    for (unsigned int i = postproc_size / 2 - 1; i < postproc_size / 2 + 2; i++) {
        for (unsigned int  j = 0; j < postproc_size; j++) {
            BYTE val = 255;
            FreeImage_SetPixelIndex(tmp, j, i, &val);
        }
    }

    for(unsigned int y = postproc_size / 2; y < height - postproc_size / 2; y++) {
        for(unsigned int x = postproc_size / 2; x < width - postproc_size / 2; x++) {
            uint8_t min = 255;
            for (unsigned int i = y - postproc_size / 2; i < y + postproc_size / 2; i++) {
                for (unsigned int j = x - postproc_size / 2; j < x + postproc_size / 2; j++) {
                    uint8_t value;
                    FreeImage_GetPixelIndex(tmp, x - j + postproc_size / 2, y - i + postproc_size / 2, &value);
                    if (value == 255) {
                        FreeImage_GetPixelIndex(image, j, i, &value);
                        min = std::min(min, value);
                    }
                }
            }
            BYTE val = min;
            FreeImage_SetPixelIndex(buffer, x, y, &val);
        }
    }

    for(unsigned int y = postproc_size / 2; y < height - postproc_size / 2; y++) {
        for(unsigned int x = postproc_size / 2; x < width - postproc_size / 2; x++) {
            uint8_t min = 0;
            for (unsigned int i = y - postproc_size / 2; i < y + postproc_size / 2; i++) {
                for (unsigned int j = x - postproc_size / 2; j < x + postproc_size / 2; j++) {
                    uint8_t value;
                    FreeImage_GetPixelIndex(tmp, x - j + postproc_size / 2, y - i + postproc_size / 2, &value);
                    if (value == 255) {
                        FreeImage_GetPixelIndex(buffer, j, i, &value);
                        min = std::max(min, value);
                    }
                }
            }
            BYTE val = min;
            FreeImage_SetPixelIndex(new_image, x, y, &val);
        }
    }

    return new_image;
}

FIBITMAP* response(FIBITMAP *image1, FIBITMAP *image2) {

    unsigned int width = FreeImage_GetWidth(image1);
    unsigned int height = FreeImage_GetHeight(image1);

    FIBITMAP *new_image = FreeImage_Allocate(width, height, 8);

    // since all pixels are unsigned, values < 0 are replaced by 0
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            uint8_t value1;
            FreeImage_GetPixelIndex(image1, x, y, &value1);
            uint8_t value2;
            FreeImage_GetPixelIndex(image2, x, y, &value2);
            int sum = value1 - value2;
            sum = std::max(0, std::min(sum, 255));

            uint8_t val = sum;
            FreeImage_SetPixelIndex(new_image, x, y, &val);
        }
    }
    return new_image;
}


FIBITMAP* threshold(FIBITMAP *image) {

    unsigned int width = FreeImage_GetWidth(image);
    unsigned int height = FreeImage_GetHeight(image);

    FIBITMAP *new_image = FreeImage_Allocate(width, height, 8);

    uint8_t maxval = 0;
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            uint8_t value;
            FreeImage_GetPixelIndex(image, x, y, &value);
            if (value > maxval) {
                maxval = value;
            }
        }
    }

    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            uint8_t value;
            FreeImage_GetPixelIndex(image, x, y, &value);
            if (value > maxval / 2) {
                BYTE val = 255;
                FreeImage_SetPixelIndex(new_image, x, y, &val);
            }
            else {
                BYTE val = 0;
                FreeImage_SetPixelIndex(new_image, x, y, &val);
            }
        }
    }
    return new_image;
}


void process_cpu(FIBITMAP *grey) {

    int pool_size = 31;
    int pp_size = 5;


    FIBITMAP* image_sobel_x = sobel(grey, 'x');
    FIBITMAP* image_sobel_y = sobel(grey, 'y');

    FIBITMAP* image_pool_sobel_x = average_pooling(image_sobel_x, pool_size);
    FIBITMAP* image_pool_sobel_y = average_pooling(image_sobel_y, pool_size);
    FIBITMAP* image_res = response(image_pool_sobel_x, image_pool_sobel_y);

    FIBITMAP* image_pp = post_processing(image_res, pp_size);
    FIBITMAP* image_output = threshold(image_pp);

    if (DEBUG) {
        FreeImage_Save(FIF_PNG, grey, "../collective_database/gray.png", 0);
        FreeImage_Save(FIF_PNG, image_sobel_x, "../collective_database/sobel_x.png", 0);
        FreeImage_Save(FIF_PNG, image_sobel_y, "../collective_database/sobel_y.png", 0);
        FreeImage_Save(FIF_PNG, image_pool_sobel_x, "../collective_database/pool_sobel_x.png", 0);
        FreeImage_Save(FIF_PNG, image_pool_sobel_y, "../collective_database/pool_sobel_y.png", 0);
        FreeImage_Save(FIF_PNG, image_res, "../collective_database/res.png", 0);
        FreeImage_Save(FIF_PNG, image_pp, "../collective_database/res_pp.png", 0);
    }

    FreeImage_Save(FIF_PNG, image_output, "../collective_database/output.png", 0);

    FreeImage_Unload(image_sobel_x);
    FreeImage_Unload(image_sobel_y);
    FreeImage_Unload(image_pool_sobel_x);
    FreeImage_Unload(image_pool_sobel_y);
    FreeImage_Unload(image_res);
    FreeImage_Unload(image_pp);
    FreeImage_Unload(image_output);

}

