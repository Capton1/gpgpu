#include "utils.hpp"

#define PNG_DEBUG 1


/*
** Compute Intersection-over-Union metric between ground truth and prediction. 
*/
float compute_IoU(Image *gt, Image *pred)
{
    float sum_inter = 0;
    float sum_union = 0;

    for(unsigned int x = 0; x < gt->width; x++) {
        for(unsigned int y = 0; y < gt->height; y++) {
            sum_inter += (gt->img[y][x] & pred->img[y][x]) != 0;
            sum_union += (gt->img[y][x] | pred->img[y][x]) != 0;
        }
    }

    return sum_inter / sum_union;
}


/*
** Scale image by $scale factor.
*/
Image* image_scaler(Image *image, float scale)
{
    unsigned int new_width = ceil(image->width * scale);
    unsigned int new_height = ceil(image->height * scale);

    Image *new_image = new Image(new_width, new_height);

    for(unsigned int x = 0; x < new_width; x++) {
        for(unsigned int y = 0; y < new_height; y++) {
            new_image->img[y][x] = image->img[(unsigned int)floor(y / scale)][(unsigned int)floor(x / scale)];
        }
    }

    return new_image;
}


/*
** Read a PNG file.
*/
Image* read_png(const char *filename)
{
    png_bytep *img;
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

    int width  = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
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

    if (img) abort();

    img = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        img[y] = (png_byte*) malloc(sizeof(png_byte) * width);
    }

    png_read_image(png, img);

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);

    Image *image = new Image(img, width, height);
    return image;
}


/*
** Write a PNG file.
*/
void write_png(Image* image, const char *filename)
{
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
            image->width, image->height,
            8,
            PNG_COLOR_TYPE_GRAY,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);


    if (!image->img) abort();

    png_write_image(png, image->img);
    png_write_end(png, NULL);

    fclose(fp);
    if (info != nullptr) png_free_data(png, info, PNG_FREE_ALL, -1);
    if (png != nullptr) png_destroy_write_struct(&png, &info);
}


/*
** Free image memory usage.
*/
void free_image(Image *image)
{
    for(unsigned int y = 0; y < image->height; y++) {
        free(image->img[y]);
    }
    free(image->img);
}