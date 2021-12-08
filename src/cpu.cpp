#include <string>
#include <stdexcept>
#include <png.h>
#include <cstring>
#include <cassert>

#define PNG_DEBUG 1


int width, height;
png_bytep *row_pointers;
void read_png_file(const char *filename) {

    png_byte color_type;
    png_byte bit_depth;

    FILE *fp = fopen(filename, "rb");

    if (fp == 0)
        abort();

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) abort();

    png_infop info = png_create_info_struct(png);

    if(!info) abort();

    if(setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth  = png_get_bit_depth(png, info);

    /* Convert 16 bit data to 8 bit */
    if (bit_depth == 16) {
        png_set_strip_16(png);
    }

    if (color_type & PNG_COLOR_MASK_ALPHA)
        png_set_strip_alpha(png);

    /* Expand to 1 pixel per byte if necessary */
    if (bit_depth < 8) png_set_packing(png);

    /* Convert to grayscale */
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_RGB_ALPHA) {

        png_set_rgb_to_gray_fixed(png, 1, -1, -1);
    }

    png_read_update_info(png, info);

    if (row_pointers) abort();

    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*) malloc(sizeof(png_byte) * width);
    }

    png_read_image(png, row_pointers);

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);

}

void write_png_file(png_bytep *src, const char *filename, int height_png, int width_png) {

    FILE *fp = fopen(filename, "wb");
    if(!fp) abort();

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
            png,
            info,
            width_png, height_png,
            8,
            PNG_COLOR_TYPE_GRAY,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);


    if (!src) abort();

    png_write_image(png, src);
    png_write_end(png, NULL);

    fclose(fp);
    if (info != nullptr) png_free_data(png, info, PNG_FREE_ALL, -1);
    if (png != nullptr) png_destroy_write_struct(&png, &info);
}


void free_image(png_bytep *src, int height) {
    for(int y = 0; y < height; y++) {
        free(src[y]);
    }
    free(src);
}

void process_png_to_sobel(png_bytep *src, png_bytep *des, const char type) {

    if (type == 'x') {

        for(int y = 1; y < height - 1; y++) {
            png_bytep row = src[y];
            png_bytep row_b = src[y - 1];
            png_bytep row_a = src[y + 1];
            for(int x = 1; x < width - 1; x++) {
                int sum = 0;

                sum += (&(row_b[x - 1]))[0] * -1;
                sum += (&(row_b[x + 1]))[0] * 1;

                sum += (&(row[x - 1]))[0] * -2;
                sum += (&(row[x + 1]))[0] * 2;

                sum += (&(row_a[x - 1]))[0] * -1;
                sum += (&(row_a[x + 1]))[0] * 1;

                sum = std::abs(sum);
                sum = std::max(0, std::min(sum, 255));

                png_byte* ptr = &(des[y][x]);
                ptr[0] = sum;
            }

        }
    }
    else if (type == 'y') {

        for(int y = 1; y < height - 1; y++) {
            png_bytep row_b = src[y - 1];
            png_bytep row_a = src[y + 1];
            for(int x = 1; x < width - 1; x++) {
                int sum = 0;

                sum += (&(row_b[x - 1]))[0] * -1;
                sum += (&(row_b[x]))[0] * -2;
                sum += (&(row_b[x + 1]))[0] * -1;

                sum += (&(row_a[x - 1]))[0] * 1;
                sum += (&(row_a[x]))[0] * 2;
                sum += (&(row_a[x + 1]))[0] * 1;

                sum = std::abs(sum);
                sum = std::max(0, std::min(sum, 255));

                png_byte* ptr = &(des[y][x]);
                ptr[0] = sum;
            }
        }
    }
    else {
        throw std::runtime_error("Invalid type");
    }
}


void average_pooling(png_bytep *src, png_bytep *des, unsigned int pool_size) {

    unsigned int patchs_y = height / pool_size;
    unsigned int patchs_x = width / pool_size;

    for (unsigned int ii = 0; ii < patchs_y; ii++) {
        for (unsigned int jj = 0; jj < patchs_x; jj++) {
            unsigned int sum = 0;
            int index_y = ii * pool_size;
            int index_x = jj * pool_size;
            for (unsigned int i = index_y; i < index_y + pool_size; i++) {
                for (unsigned int j = index_x; j < index_x + pool_size; j++) {
                    sum += (&(src[i][j]))[0];
                }
            }
            (&(des[ii][jj]))[0] = sum / (pool_size * pool_size);
        }
    }
}

void post_processing(png_bytep *src, png_bytep *des, unsigned int postproc_size, int patch_y, int patch_x) {

    png_bytep *se = (png_bytep *) malloc(sizeof(png_bytep) * postproc_size);
    for (unsigned int y = 0; y < postproc_size; y++) {
        se[y] = (png_byte *) calloc(postproc_size, sizeof(png_byte));
    }

    for (unsigned int i = postproc_size / 2 - 1; i < postproc_size / 2 + 2; i++) {
        for (unsigned int  j = 0; j < postproc_size; j++) {
            (&(se[i][j]))[0] = 255;
        }
    }

    for(unsigned int y = postproc_size / 2; y < patch_y - postproc_size / 2; y++) {
        for(unsigned int x = postproc_size / 2; x < patch_x - postproc_size / 2; x++) {
            int min = 255;
            for (unsigned int i = y - postproc_size / 2; i < y + postproc_size / 2; i++) {
                for (unsigned int j = x - postproc_size / 2; j < x + postproc_size / 2; j++) {
                    if ((&(se[y - i + postproc_size / 2][x - j + postproc_size / 2]))[0] == 255) {
                        int curr = (&(src[i][j]))[0];
                        min = std::min(min, curr);
                    }
                }
            }
            (&(des[y][x]))[0] = min;
        }
    }

    for(unsigned int y = postproc_size / 2; y < patch_y - postproc_size / 2; y++) {
        for (unsigned int x = postproc_size / 2; x < patch_x - postproc_size / 2; x++) {
            (&(src[y][x]))[0] = (&(des[y][x]))[0];
        }
    }


    for(unsigned int y = postproc_size / 2; y < patch_y - postproc_size / 2; y++) {
        for(unsigned int x = postproc_size / 2; x < patch_x - postproc_size / 2; x++) {
            int min = 0;
            for (unsigned int i = y - postproc_size / 2; i < y + postproc_size / 2; i++) {
                for (unsigned int j = x - postproc_size / 2; j < x + postproc_size / 2; j++) {
                    if ((&(se[y - i + postproc_size / 2][x - j + postproc_size / 2]))[0] == 255) {
                        int curr = (&(src[i][j]))[0];
                        min = std::max(min, curr);
                    }
                }
            }
            (&(des[y][x]))[0] = min;
        }
    }

    free_image(se, postproc_size);
}

void response(png_bytep *src1, png_bytep *src2, png_bytep *des, unsigned int patch_y, unsigned int patch_x) {
    // since all pixels are unsigned, values < 0 are replaced by 0
    for (unsigned int y = 0; y < patch_y; y++) {
        for (unsigned int x = 0; x < patch_x; x++) {
            int sum = (&(src1[y][x]))[0] - (&(src2[y][x]))[0];
            sum = std::max(0, std::min(sum, 255));
            (&(des[y][x]))[0] = sum;
        }
    }
}


void threshold(png_bytep *src, png_bytep *des, unsigned int patch_y, unsigned int patch_x) {
    char maxval = 0;
    for (unsigned int y = 0; y < patch_y; y++) {
        for (unsigned int x = 0; x < patch_x; x++) {
            char curr = (&(src[y][x]))[0];
            if (curr > maxval) {
                maxval = curr;
            }
        }
    }

    for (unsigned int y = 0; y < patch_y; y++) {
        for (unsigned int x = 0; x < patch_x; x++) {
            char curr = (&(src[y][x]))[0];
            if (curr > maxval / 2) {
                (&(des[y][x]))[0] = 255;
            }
            else {
                (&(des[y][x]))[0] = 0;
            }
        }
    }
}

void process_cpu(const char* img_path) {
    std::string str;

    read_png_file(img_path);
    write_png_file(row_pointers, "../collective_database/gray.png", height, width);

    // SobelX
    png_bytep *sobx1 = (png_bytep *) malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        sobx1[y] = (png_byte *) malloc(sizeof(png_byte) * width);
    }

    process_png_to_sobel(row_pointers, sobx1, 'x');
    write_png_file(sobx1, "../collective_database/sobelX.png", height, width);

    // SobelY
    png_bytep *soby1 = (png_bytep *) malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        soby1[y] = (png_byte *) malloc(sizeof(png_byte) * width);
    }

    process_png_to_sobel(row_pointers, soby1, 'y');
    write_png_file(soby1, "../collective_database/sobelY.png", height, width);


    int pool_size = 31;
    int patchs_y = height / pool_size;
    int patchs_x = width / pool_size;

    png_bytep *pool_sobx1 = (png_bytep *) malloc(sizeof(png_bytep) * patchs_y);
    png_bytep *pool_soby1 = (png_bytep *) malloc(sizeof(png_bytep) * patchs_y);
    for (int y = 0; y < patchs_y; y++) {
        pool_sobx1[y] = (png_byte *) malloc(sizeof(png_byte) * patchs_x);
        pool_soby1[y] = (png_byte *) malloc(sizeof(png_byte) * patchs_x);
    }


    average_pooling(sobx1, pool_sobx1, pool_size);
    write_png_file(pool_sobx1, "../collective_database/pool_sobx1.png", patchs_y, patchs_x);

    average_pooling(soby1, pool_soby1, pool_size);
    write_png_file(pool_soby1, "../collective_database/pool_soby1.png", patchs_y, patchs_x);


    png_bytep *resp1 = (png_bytep *) malloc(sizeof(png_bytep) * patchs_y);
    for (int y = 0; y < patchs_y; y++) {
        resp1[y] = (png_byte *) malloc(sizeof(png_byte) * patchs_x);
    }

    response(pool_sobx1, pool_soby1, resp1, patchs_y, patchs_x);
    write_png_file(resp1, "../collective_database/resp1.png", patchs_y, patchs_x);


    int postproc_size = 5;
    png_bytep *resp1_postproc = (png_bytep *) malloc(sizeof(png_bytep) * patchs_y);
    for (int y = 0; y < patchs_y; y++) {
        resp1_postproc[y] = (png_byte *) calloc(patchs_x, sizeof(png_byte));
    }

    post_processing(resp1, resp1_postproc, postproc_size, patchs_y, patchs_x);
    write_png_file(resp1_postproc, "../collective_database/resp1_postproc.png", patchs_y, patchs_x);

    png_bytep *output = (png_bytep *) malloc(sizeof(png_bytep) * patchs_y);
    for (int y = 0; y < patchs_y; y++) {
        output[y] = (png_byte *) malloc(sizeof(png_byte) * patchs_x);
    }

    threshold(resp1_postproc, output, patchs_y, patchs_x);
    write_png_file(output, "../collective_database/output.png", patchs_y, patchs_x);

    free_image(row_pointers, height);
    free_image(sobx1, height);
    free_image(soby1, height);
    free_image(pool_sobx1, patchs_y);
    free_image(pool_soby1, patchs_y);
    free_image(resp1, patchs_y);
    free_image(resp1_postproc, patchs_y);
    free_image(output, patchs_y);

}

