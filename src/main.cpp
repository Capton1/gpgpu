#include <CLI/CLI.hpp>
#include<vector>
#include<algorithm>
#include "process.hpp"
#include <stdio.h>
#include "cpu.hpp"
#include <FreeImage.h>


int main(int argc, char** argv) {
    (void) argc;
    (void) argv;

    std::string filename = "../collective_database/test.png";

    
    std::string mode = "CPU";

    CLI::App app{"code"};
    app.add_option("-i", filename, "Input image");
    app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");
    CLI11_PARSE(app, argc, argv);

    const char *filename_c = filename.c_str();
    FREE_IMAGE_FORMAT formato = FreeImage_GetFileType(filename_c, 0);
    FIBITMAP *src = FreeImage_Load(formato, filename_c);
    FIBITMAP *grey;

    grey = FreeImage_ConvertToGreyscale(src);

    int width = FreeImage_GetWidth(src);
    int height = FreeImage_GetHeight(src);

    // Rendering
    if (mode == "CPU") {
        printf("CPU\n");
        //process_cpu(grey);
    }
    else if (mode == "GPU") {
        int output_width = width/POOLSIZE;
        int output_height = height/POOLSIZE;

        uint8_t* buffer = (uint8_t*) FreeImage_GetBits(grey);

        uint8_t *output = (uint8_t*)calloc(output_width * output_height, sizeof(uint8_t));
        process_image(buffer, output, width, height);

        auto output_img = FreeImage_ConvertFromRawBits(output, output_width, output_height,
                                                        output_width, 8, 0, 0, 0);

        FreeImage_Save(FIF_PNG, output_img, "../output_gpu/output.png", 0);
        free(output);
    }


    FreeImage_Unload(src);
    FreeImage_Unload(grey);

    // Evaluate
    /*int pool_size = 31;

    Image* output = read_png("../collective_database/output.png");
    Image* scaled_output = image_scaler(output, pool_size+1);
    write_png(scaled_output, "../collective_database/scaled_output.png");

    //Image* gt = read_png("../collective_database/test-GT.png");
    
    //float iou = compute_IoU(gt, scaled_output);
    //printf("Metrics: IoU: %f\n", iou);*/
}
