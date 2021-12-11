#include "image.hpp"

Image::Image(png_bytep* _img, unsigned int _width, unsigned int _height)
    : img(_img), width(_width), height(_height)
{

}

Image::Image(unsigned int _width, unsigned int _height)
    : width(_width), height(_height)
{
    this->img = (png_bytep *) malloc(sizeof(png_bytep) * _height);
    for (unsigned int y = 0; y < _height; y++) {
        this->img[y] = (png_byte *) calloc(_width, sizeof(png_byte));
    }
}

Image::~Image()
{
    for(unsigned int y = 0; y < this->height; y++) {
        free(this->img[y]);
    }
    free(this->img);
}