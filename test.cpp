#include <iostream>
#include <FreeImage.h>

int main()
{
    FREE_IMAGE_FORMAT formato = FreeImage_GetFileType("collective_database/test.png", 0);
    FIBITMAP *src = FreeImage_Load(formato, "collective_database/test.png");
    FIBITMAP *grey;

    grey = FreeImage_ConvertToGreyscale(src);

    //int scan_width = FreeImage_GetPitch(src);
    //BYTE *bits = (BYTE*)malloc(height * scan_width);
    //FreeImage_ConvertToRawBits(bits, src, scan_width, 32, 0, 0, 0, 0);

    int width = FreeImage_GetWidth(src);
    int height = FreeImage_GetHeight(src);

    char* pixeles = (char*) FreeImage_GetBits(grey);
    for(int j= 0; j < width * 50; j++)
    {
        pixeles[j] = 0;
    }

    FreeImage_Save(FIF_PNG, grey, "mybitmap.png", 0);
    FreeImage_Unload(grey);
    FreeImage_Unload(src);
    return 0;
}