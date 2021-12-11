#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <png.h>
#include <stdlib.h>

class Image
{
    public:
    Image(png_bytep* _img, unsigned int _width, unsigned int _height);
    Image(unsigned int _width, unsigned int _height);
    ~Image();
    
    png_bytep* img;
    unsigned int width;
    unsigned int height;
};


#endif