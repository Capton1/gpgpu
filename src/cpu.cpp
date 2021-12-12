#include "cpu.hpp"


Image* sobel(Image *image, const char type) {
    Image *new_image = new Image(image->width, image->height);

    if (type == 'x') {

        for(unsigned int y = 1; y < image->height - 1; y++) {
            png_bytep row = image->img[y];
            png_bytep row_b = image->img[y - 1];
            png_bytep row_a = image->img[y + 1];
            for(unsigned int x = 1; x < image->width - 1; x++) {
                int sum = 0;

                sum += (&(row_b[x - 1]))[0] * -1;
                sum += (&(row_b[x + 1]))[0] * 1;

                sum += (&(row[x - 1]))[0] * -2;
                sum += (&(row[x + 1]))[0] * 2;

                sum += (&(row_a[x - 1]))[0] * -1;
                sum += (&(row_a[x + 1]))[0] * 1;

                sum = std::abs(sum);
                sum = std::max(0, std::min(sum, 255));

                png_byte* ptr = &(new_image->img[y][x]);
                ptr[0] = sum;
            }

        }
    }
    else if (type == 'y') {

        for(unsigned int y = 1; y < image->height - 1; y++) {
            png_bytep row_b = image->img[y - 1];
            png_bytep row_a = image->img[y + 1];
            for(unsigned int x = 1; x < image->width - 1; x++) {
                int sum = 0;

                sum += (&(row_b[x - 1]))[0] * -1;
                sum += (&(row_b[x]))[0] * -2;
                sum += (&(row_b[x + 1]))[0] * -1;

                sum += (&(row_a[x - 1]))[0] * 1;
                sum += (&(row_a[x]))[0] * 2;
                sum += (&(row_a[x + 1]))[0] * 1;

                sum = std::abs(sum);
                sum = std::max(0, std::min(sum, 255));

                png_byte* ptr = &(new_image->img[y][x]);
                ptr[0] = sum;
            }
        }
    }
    else {
        throw std::runtime_error("Invalid type");
    }

    return new_image;
}


Image* average_pooling(Image *image, unsigned int pool_size) {
    Image *new_image = new Image(image->width / pool_size, image->height / pool_size);

    for (unsigned int ii = 0; ii < new_image->height; ii++) {
        for (unsigned int jj = 0; jj < new_image->width; jj++) {
            unsigned int sum = 0;
            int index_y = ii * pool_size;
            int index_x = jj * pool_size;
            for (unsigned int i = index_y; i < index_y + pool_size; i++) {
                for (unsigned int j = index_x; j < index_x + pool_size; j++) {
                    sum += (&(image->img[i][j]))[0];
                }
            }
            (&(new_image->img[ii][jj]))[0] = sum / (pool_size * pool_size);
        }
    }
    
    return new_image;
}


Image* post_processing(Image *image, unsigned int postproc_size) {
    Image *new_image = new Image(image->width, image->height);
    Image *tmp = new Image(postproc_size, postproc_size);

    for (unsigned int i = postproc_size / 2 - 1; i < postproc_size / 2 + 2; i++) {
        for (unsigned int  j = 0; j < postproc_size; j++) {
            (&(tmp->img[i][j]))[0] = 255;
        }
    }

    for(unsigned int y = postproc_size / 2; y < image->height - postproc_size / 2; y++) {
        for(unsigned int x = postproc_size / 2; x < image->width - postproc_size / 2; x++) {
            int min = 255;
            for (unsigned int i = y - postproc_size / 2; i < y + postproc_size / 2; i++) {
                for (unsigned int j = x - postproc_size / 2; j < x + postproc_size / 2; j++) {
                    if ((&(tmp->img[y - i + postproc_size / 2][x - j + postproc_size / 2]))[0] == 255) {
                        int curr = (&(image->img[i][j]))[0];
                        min = std::min(min, curr);
                    }
                }
            }
            (&(new_image->img[y][x]))[0] = min;
        }
    }

    for(unsigned int y = postproc_size / 2; y < image->height - postproc_size / 2; y++) {
        for (unsigned int x = postproc_size / 2; x < image->width - postproc_size / 2; x++) {
            (&(image->img[y][x]))[0] = (&(new_image->img[y][x]))[0];
        }
    }


    for(unsigned int y = postproc_size / 2; y < image->height - postproc_size / 2; y++) {
        for(unsigned int x = postproc_size / 2; x < image->width - postproc_size / 2; x++) {
            int min = 0;
            for (unsigned int i = y - postproc_size / 2; i < y + postproc_size / 2; i++) {
                for (unsigned int j = x - postproc_size / 2; j < x + postproc_size / 2; j++) {
                    if ((&(tmp->img[y - i + postproc_size / 2][x - j + postproc_size / 2]))[0] == 255) {
                        int curr = (&(image->img[i][j]))[0];
                        min = std::max(min, curr);
                    }
                }
            }
            (&(new_image->img[y][x]))[0] = min;
        }
    }

    return new_image;
}


Image* response(Image *image1, Image *image2) {
    Image *new_image = new Image(image1->width, image1->height);
    // since all pixels are unsigned, values < 0 are replaced by 0
    for (unsigned int y = 0; y < image1->height; y++) {
        for (unsigned int x = 0; x < image1->width; x++) {
            int sum = (&(image1->img[y][x]))[0] - (&(image2->img[y][x]))[0];
            sum = std::max(0, std::min(sum, 255));
            (&(new_image->img[y][x]))[0] = sum;
        }
    }
    return new_image;
}


Image* threshold(Image *image) {
    Image *new_image = new Image(image->width, image->height);

    char maxval = 0;
    for (unsigned int y = 0; y < image->height; y++) {
        for (unsigned int x = 0; x < image->width; x++) {
            char curr = (&(image->img[y][x]))[0];
            if (curr > maxval) {
                maxval = curr;
            }
        }
    }

    for (unsigned int y = 0; y < image->height; y++) {
        for (unsigned int x = 0; x < image->width; x++) {
            char curr = (&(image->img[y][x]))[0];
            if (curr > maxval / 2) {
                (&(new_image->img[y][x]))[0] = 255;
            }
            else {
                (&(new_image->img[y][x]))[0] = 0;
            }
        }
    }

    return new_image;
}


void process_cpu(const char* img_path) {

    Image* image = read_png(img_path);
    write_png(image, "../collective_database/gray.png");

    Image* image_sobel_x = sobel(image, 'x');
    write_png(image_sobel_x, "../collective_database/sobel_x.png");

    Image* image_sobel_y = sobel(image, 'y');
    write_png(image_sobel_y, "../collective_database/sobel_y.png");

    int pool_size = 31;

    Image* image_pool_sobel_x = average_pooling(image_sobel_x, pool_size);
    write_png(image_pool_sobel_x, "../collective_database/pool_sobel_x.png");

    Image* image_pool_sobel_y = average_pooling(image_sobel_y, pool_size);
    write_png(image_pool_sobel_y, "../collective_database/pool_sobel_y.png");


    Image* image_res = response(image_pool_sobel_x, image_pool_sobel_y);
    write_png(image_res, "../collective_database/res.png");

    int pp_size = 5;

    Image* image_pp = post_processing(image_res, pp_size);
    write_png(image_pp, "../collective_database/res_pp.png");

    Image* image_output = threshold(image_pp);
    write_png(image_output, "../collective_database/output.png");
}

