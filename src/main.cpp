#include <CLI/CLI.hpp>
#include<vector>
#include<algorithm>
#include "process.hpp"
#include <stdio.h>
#include "cpu.hpp"
#include "utils.hpp"
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
        process_cpu(grey);
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
    int pool_size = 31;

    formato = FreeImage_GetFileType("../collective_database/output.png", 0);
    FIBITMAP *output = FreeImage_Load(formato, "../collective_database/output.png");
    FIBITMAP *output_grey = FreeImage_ConvertToGreyscale(output);
    FIBITMAP *scaled_output = image_scaler(output, pool_size + 1);
    FreeImage_Save(FIF_PNG, scaled_output, "../collective_database/scaled_output.png", 0);
    formato = FreeImage_GetFileType("../collective_database/test-GT.png", 0);
    FIBITMAP *gt = FreeImage_Load(formato, "../collective_database/test-GT.png");
    FIBITMAP *gt_grey = FreeImage_ConvertToGreyscale(gt);
    float iou = compute_IoU(gt_grey, scaled_output);
    printf("Metrics: IoU: %f\n", iou);

    FreeImage_Unload(gt);
    FreeImage_Unload(gt_grey);
    FreeImage_Unload(output);
    FreeImage_Unload(output_grey);
}
